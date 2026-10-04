# -*- coding: utf-8 -*-
"""
Interface unificada para diagramas ternarios de ELL.

Consolida os quatro modulos do projeto numa unica aplicacao:
  unifac_ll.py        -> modelo UNIFAC-LL
  cosmo_sac_adapter.py-> modelo COSMO-SAC (via JCOSMO/py4j)
  thermo_engine.py    -> estabilidade, flash e varredura da grade
  ternary_diagram.py  -> renderizacao e exportacoes

Execucao:  python interface_ell.py

A logica de calculo (compute_equilibrium, check_parameters, render_and_export)
e independente do Tkinter e pode ser chamada de scripts/testes. A GUI apenas
coleta os valores dos widgets e dispara essa logica numa thread de fundo,
comunicando-se por uma fila thread-safe. Todo o matplotlib roda na thread
principal.
"""

import time
import queue
import threading

from unifac_ll import UnifacLL, COMPONENT_LABEL
from cosmo_sac_adapter import CosmoSacAdapter, CosmoSacError
from thermo_engine import grid_scan
from ternary_diagram import plot_ternary


# ---------------------------------------------------------------------------
# Tabelas de selecao (rotulo exibido -> chave interna)
# ---------------------------------------------------------------------------
OIL_OPTIONS = [("Trioleina", "trioleina"),
               ("Trilinoleina", "trilinoleina"),
               ("Trilinolenina", "trilinolenina"),
               ("Triricinoleina", "triricinoleina"),
               ("Tripalmitina", "tripalmitina"),
               ("Triestearina", "triestearina"),
               ("Acido oleico", "acido_oleico"),
               ("Acido linoleico", "acido_linoleico"),
               ("Acido ricinoleico", "acido_ricinoleico")]

ALCOHOL_OPTIONS = [("Metanol", "metanol"),
                   ("Etanol", "etanol"),
                   ("1-Propanol", "1-propanol")]

IL_OPTIONS = [("[2-HEA][Hx]", "2HEAHX"),
              ("[DETA][Hx]", "DETAHX")]

MODEL_OPTIONS = ["UNIFAC-LL", "COSMO-SAC"]

# nomes padrao dos compostos no banco do JCOSMO (editaveis na interface).
# Trilinoleina e trilinolenina foram confirmadas com perfil sigma existente
# no banco (a autora obteve gamma para ambas). Triricinoleina, tripalmitina
# e triestearina AINDA NAO tem perfil sigma/segments completo no banco no
# momento desta atualizacao (erro relatado ao consultar o JCOSMO); os nomes
# abaixo sao placeholders para quando esses perfis existirem.
JCOSMO_DEFAULTS = {
    "trioleina": "TRIOLEIN",
    "trilinoleina": "TRILINOLEIN",
    "trilinolenina": "TRILINOLENIN",
    "triricinoleina": "TRIRICINOLEIN",   # sem SEGMENTS no banco ainda
    "tripalmitina": "TRIPALMITIN",       # sem SEGMENTS no banco ainda
    "triestearina": "TRIESTEARIN",       # sem perfil sigma no banco ainda
    "acido_oleico": "OLEIC_ACID",
    "acido_linoleico": "LINOLEIC_ACID",
    "acido_ricinoleico": "RICINOLEIC_ACID",
    "metanol": "METHANOL",
    "etanol": "ETHANOL",
    "1-propanol": "1-PROPANOL",
    "2HEAHX": "2HEAHX",
    "DETAHX": "[DETA][HX]",
}


def _labels_for(comp_keys):
    return [COMPONENT_LABEL[k] for k in comp_keys]


# ---------------------------------------------------------------------------
# Logica de calculo (independente de GUI; testavel diretamente)
# ---------------------------------------------------------------------------
def check_parameters(comp_keys, log_cb):
    """Relatorio de pares de interacao ausentes (UNIFAC-LL)."""
    model = UnifacLL(comp_keys, 298.15)
    log_cb(model.missing_pairs_report())
    if model.missing_pairs():
        log_cb(">> Resultado seria quantitativamente NAO CONFIAVEL "
               "para este sistema.")
    return model.missing_pairs()


def compute_equilibrium(comp_keys, model_choice, T, n_grid,
                        jcosmo_names=None, log_cb=None, progress_cb=None):
    """Constroi o modelo e faz a varredura. NAO renderiza (sem matplotlib),
    para poder rodar em thread de fundo com seguranca. Retorna dict com as
    tie-lines e metadados."""
    log = log_cb or (lambda s: None)

    if model_choice == "UNIFAC-LL":
        model = UnifacLL(comp_keys, T)
        log(model.missing_pairs_report())
        unreliable = bool(model.missing_pairs())
        if unreliable:
            log(">> AVISO: pares de interacao ausentes. Resultado "
                "quantitativamente NAO CONFIAVEL (calculo segue com Psi=1).")
    else:
        log("Conectando ao JCOSMO (py4j)...")
        model = CosmoSacAdapter(jcosmo_names, T)   # pode lancar CosmoSacError
        log("Conexao estabelecida. Modelo: %s" % model.model_name)
        unreliable = False

    log("Varredura da grade (n_grid=%d)..." % n_grid)
    t0 = time.time()
    out = grid_scan(model, n_grid=n_grid,
                    progress_callback=progress_cb, log_callback=log)
    elapsed = time.time() - t0

    out["elapsed"] = elapsed
    out["jcosmo_calls"] = getattr(model, "n_calls", None)
    out["unreliable"] = unreliable
    out["comp_keys"] = list(comp_keys)
    out["model_choice"] = model_choice
    out["model"] = model   # usado pelo solver de ponto critico na renderizacao
    return out


def render_and_export(out, comp_labels, return_fig=False, outdir=None):
    """Renderiza o diagrama a partir do resultado da varredura e exporta
    PNG/CSV. Deve rodar na thread principal (matplotlib). Se outdir for
    dado, salva nele (usado pela fila em lote); senao usa o padrao."""
    system_name = " + ".join(comp_labels)
    kw = dict(system_name=system_name,
              jcosmo_calls=out.get("jcosmo_calls"),
              elapsed=out.get("elapsed"),
              return_fig=return_fig,
              model=out.get("model"))
    if outdir is not None:
        kw["outdir"] = outdir
    return plot_ternary(out["tie_lines"], comp_labels, out["model_choice"], **kw)


# ---------------------------------------------------------------------------
# GUI Tkinter
# ---------------------------------------------------------------------------
def main(_test_hook=None):
    import tkinter as tk
    from tkinter import ttk
    from tkinter.scrolledtext import ScrolledText

    class EllApp:
        def __init__(self, root):
            self.root = root
            root.title("Diagramas Ternarios de ELL - UNIFAC-LL / COSMO-SAC")
            self.queue = queue.Queue()
            self._fig = None
            self._canvas = None

            main = ttk.Frame(root, padding=10)
            main.grid(row=0, column=0, sticky="nsew")
            root.columnconfigure(0, weight=1)
            root.rowconfigure(0, weight=1)

            # ----- painel de selecao (esquerda) -----
            ctrl = ttk.LabelFrame(main, text="Configuracao", padding=8)
            ctrl.grid(row=0, column=0, sticky="nw", padx=(0, 10))

            self.oil_var = tk.StringVar(value=OIL_OPTIONS[0][0])
            self.alc_var = tk.StringVar(value=ALCOHOL_OPTIONS[1][0])  # Etanol
            self.il_var = tk.StringVar(value=IL_OPTIONS[0][0])
            self.model_var = tk.StringVar(value=MODEL_OPTIONS[0])
            self.T_var = tk.StringVar(value="298.15")
            self.ngrid_var = tk.StringVar(value="20")

            r = 0
            ttk.Label(ctrl, text="Oleo / acido graxo:").grid(row=r, column=0, sticky="w")
            r += 1
            ttk.Combobox(ctrl, textvariable=self.oil_var, state="readonly",
                         values=[o[0] for o in OIL_OPTIONS], width=24
                         ).grid(row=r, column=0, sticky="ew", pady=(0, 6))
            r += 1
            ttk.Label(ctrl, text="Alcool:").grid(row=r, column=0, sticky="w")
            r += 1
            ttk.Combobox(ctrl, textvariable=self.alc_var, state="readonly",
                         values=[o[0] for o in ALCOHOL_OPTIONS], width=24
                         ).grid(row=r, column=0, sticky="ew", pady=(0, 6))
            r += 1
            ttk.Label(ctrl, text="Liquido ionico / cossolvente:").grid(row=r, column=0, sticky="w")
            r += 1
            ttk.Combobox(ctrl, textvariable=self.il_var, state="readonly",
                         values=[o[0] for o in IL_OPTIONS], width=24
                         ).grid(row=r, column=0, sticky="ew", pady=(0, 6))
            r += 1

            # modelo (radio)
            mfr = ttk.LabelFrame(ctrl, text="Modelo", padding=4)
            mfr.grid(row=r, column=0, sticky="ew", pady=6)
            for i, mo in enumerate(MODEL_OPTIONS):
                ttk.Radiobutton(mfr, text=mo, value=mo, variable=self.model_var,
                                command=self._on_model_change
                                ).grid(row=0, column=i, sticky="w", padx=4)
            r += 1

            # campos JCOSMO (aparecem no COSMO-SAC)
            self.jcosmo_frame = ttk.LabelFrame(
                ctrl, text="Nomes no banco do JCOSMO (exatos)", padding=4)
            self.jcosmo_frame.grid(row=r, column=0, sticky="ew", pady=6)
            self.jcosmo_vars = [tk.StringVar(), tk.StringVar(), tk.StringVar()]
            self.jcosmo_labels = []
            for i in range(3):
                lab = ttk.Label(self.jcosmo_frame, text="")
                lab.grid(row=i, column=0, sticky="w")
                self.jcosmo_labels.append(lab)
                ttk.Entry(self.jcosmo_frame, textvariable=self.jcosmo_vars[i],
                          width=22).grid(row=i, column=1, sticky="ew", padx=4, pady=1)
            r += 1

            # parametros numericos
            pfr = ttk.Frame(ctrl)
            pfr.grid(row=r, column=0, sticky="ew", pady=4)
            ttk.Label(pfr, text="T (K):").grid(row=0, column=0, sticky="w")
            ttk.Entry(pfr, textvariable=self.T_var, width=10).grid(row=0, column=1, padx=4)
            ttk.Label(pfr, text="n_grid:").grid(row=0, column=2, sticky="w", padx=(8, 0))
            ttk.Entry(pfr, textvariable=self.ngrid_var, width=6).grid(row=0, column=3, padx=4)
            r += 1

            # botoes
            bfr = ttk.Frame(ctrl)
            bfr.grid(row=r, column=0, sticky="ew", pady=(8, 4))
            self.btn_check = ttk.Button(bfr, text="Verificar parametros",
                                        command=self._on_check)
            self.btn_check.grid(row=0, column=0, padx=2)
            self.btn_calc = ttk.Button(bfr, text="Calcular diagrama",
                                       command=self._on_calc)
            self.btn_calc.grid(row=0, column=1, padx=2)
            self.btn_fila = ttk.Button(bfr, text="Rodar fila (CSV)",
                                       command=self._on_fila)
            self.btn_fila.grid(row=0, column=2, padx=2)
            self.btn_parar = ttk.Button(bfr, text="Parar fila",
                                        command=self._on_parar_fila,
                                        state="disabled")
            self.btn_parar.grid(row=0, column=3, padx=2)
            r += 1

            self.progress = ttk.Progressbar(ctrl, mode="determinate", length=240)
            self.progress.grid(row=r, column=0, sticky="ew", pady=4)

            # ----- painel de saida (direita): diagrama + log -----
            right = ttk.Frame(main)
            right.grid(row=0, column=1, sticky="nsew")
            main.columnconfigure(1, weight=1)
            main.rowconfigure(0, weight=1)

            self.plot_frame = ttk.LabelFrame(right, text="Diagrama", padding=4)
            self.plot_frame.grid(row=0, column=0, sticky="nsew")
            right.rowconfigure(0, weight=1)
            right.columnconfigure(0, weight=1)

            logfr = ttk.LabelFrame(right, text="Log", padding=4)
            logfr.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
            right.rowconfigure(1, weight=1)
            self.log = ScrolledText(logfr, height=10, width=70, wrap="word",
                                    font=("TkDefaultFont", 9))
            self.log.grid(row=0, column=0, sticky="nsew")
            logfr.rowconfigure(0, weight=1)
            logfr.columnconfigure(0, weight=1)

            self._on_model_change()
            self._poll_queue()

        # ---- utilidades de estado ----
        def _keys(self):
            oil = dict(OIL_OPTIONS)[self.oil_var.get()]
            alc = dict(ALCOHOL_OPTIONS)[self.alc_var.get()]
            il = dict(IL_OPTIONS)[self.il_var.get()]
            return [oil, alc, il]

        def _log_write(self, text):
            self.log.insert("end", text + "\n")
            self.log.see("end")

        def _on_model_change(self):
            keys = self._keys()
            is_cosmo = self.model_var.get() == "COSMO-SAC"
            state = "normal" if is_cosmo else "disabled"
            # preenche defaults e habilita/desabilita campos JCOSMO
            for i, k in enumerate(keys):
                self.jcosmo_labels[i].config(text=_labels_for([k])[0] + ":")
                if is_cosmo and not self.jcosmo_vars[i].get():
                    self.jcosmo_vars[i].set(JCOSMO_DEFAULTS[k])
            for child in self.jcosmo_frame.winfo_children():
                try:
                    child.config(state=state)
                except tk.TclError:
                    pass
            self.btn_check.config(
                state="disabled" if is_cosmo else "normal")

        def _refresh_jcosmo_defaults(self):
            for i, k in enumerate(self._keys()):
                self.jcosmo_vars[i].set(JCOSMO_DEFAULTS[k])

        # ---- acoes dos botoes ----
        def _on_check(self):
            self._log_write("")
            keys = self._keys()
            self._log_write("Sistema: " + " + ".join(_labels_for(keys)))
            try:
                check_parameters(keys, self._log_write)
            except Exception as exc:
                self._log_write("Erro: %s" % exc)

        def _on_calc(self):
            keys = self._keys()
            model_choice = self.model_var.get()
            try:
                T = float(self.T_var.get())
                n_grid = int(self.ngrid_var.get())
            except ValueError:
                self._log_write("Erro: T e n_grid devem ser numericos.")
                return
            jcosmo_names = None
            if model_choice == "COSMO-SAC":
                self._refresh_missing_jcosmo(keys)
                jcosmo_names = [v.get() for v in self.jcosmo_vars]

            self.btn_calc.config(state="disabled")
            self.btn_check.config(state="disabled")
            self.progress.config(value=0)
            self._log_write("")
            self._log_write("=== %s ===" % " + ".join(_labels_for(keys)))

            args = (keys, model_choice, T, n_grid, jcosmo_names)
            threading.Thread(target=self._worker, args=args, daemon=True).start()

        def _refresh_missing_jcosmo(self, keys):
            for i, k in enumerate(keys):
                if not self.jcosmo_vars[i].get():
                    self.jcosmo_vars[i].set(JCOSMO_DEFAULTS[k])

        def _on_fila(self):
            from tkinter import filedialog, messagebox
            csv_path = filedialog.askopenfilename(
                title="Escolha o CSV da fila",
                filetypes=[("CSV", "*.csv"), ("Todos", "*.*")])
            if not csv_path:
                return
            outdir = filedialog.askdirectory(
                title="Pasta de saida da fila (cancele para usar 'resultados_fila')")
            if not outdir:
                outdir = "resultados_fila"

            self._fila_parar = False
            self.btn_calc.config(state="disabled")
            self.btn_check.config(state="disabled")
            self.btn_fila.config(state="disabled")
            self.btn_parar.config(state="normal")
            self.progress.config(mode="indeterminate")
            self.progress.start(12)
            self._log_write("")
            self._log_write("=== FILA: %s ===" % csv_path)

            args = (csv_path, outdir)
            threading.Thread(target=self._fila_worker, args=args,
                             daemon=True).start()

        def _on_parar_fila(self):
            # sinaliza a parada; a fila termina o sistema em andamento e
            # entao para (a flag e lida por stop_flag no run_batch).
            self._fila_parar = True
            self.btn_parar.config(state="disabled")
            self._log_write(">> Parada solicitada. A fila vai encerrar apos "
                            "concluir o sistema em andamento...")

        def _fila_worker(self, csv_path, outdir):
            from batch_runner import run_batch

            def q_log(s):
                self.queue.put(("log", s))

            def q_prog(i, n, nome):
                self.queue.put(("log", ">> [%d/%d] iniciando: %s" % (i, n, nome)))

            try:
                run_batch(csv_path, outdir=outdir, log_cb=q_log,
                          progress_cb=q_prog,
                          stop_flag=lambda: getattr(self, "_fila_parar", False))
                self.queue.put(("fila_done", outdir))
            except Exception as exc:
                self.queue.put(("error", "Erro na fila: %s" % exc))
                self.queue.put(("fila_done", outdir))

        # ---- thread de fundo ----
        def _worker(self, keys, model_choice, T, n_grid, jcosmo_names):
            def q_log(s):
                self.queue.put(("log", s))

            def q_prog(i, total):
                self.queue.put(("progress", (i, total)))

            try:
                out = compute_equilibrium(
                    keys, model_choice, T, n_grid, jcosmo_names,
                    log_cb=q_log, progress_cb=q_prog)
                self.queue.put(("computed", out))
            except CosmoSacError as exc:
                self.queue.put(("error", str(exc)))
            except Exception as exc:
                self.queue.put(("error", "Erro no calculo: %s" % exc))

        # ---- consumo da fila (thread principal) ----
        def _poll_queue(self):
            try:
                while True:
                    kind, payload = self.queue.get_nowait()
                    if kind == "log":
                        self._log_write(payload)
                    elif kind == "progress":
                        i, total = payload
                        self.progress.config(maximum=total, value=i)
                    elif kind == "computed":
                        self._on_computed(payload)
                    elif kind == "error":
                        self._log_write(">> " + payload)
                        self.btn_calc.config(state="normal")
                        self._on_model_change()
                    elif kind == "fila_done":
                        self.progress.stop()
                        self.progress.config(mode="determinate", value=0)
                        self.btn_calc.config(state="normal")
                        self.btn_check.config(state="normal")
                        self.btn_fila.config(state="normal")
                        self.btn_parar.config(state="disabled")
                        self._log_write("Fila finalizada. Resultados em: %s"
                                        % payload)
                        self._on_model_change()
            except queue.Empty:
                pass
            self.root.after(100, self._poll_queue)

        def _on_computed(self, out):
            comp_labels = _labels_for(out["comp_keys"])
            png, csv, plait, fig = render_and_export(out, comp_labels, return_fig=True)
            self._log_write("Concluido. Tie-lines aceitas: %d." % out["n_accepted"])
            self._log_write("PNG: %s" % png)
            if csv:
                self._log_write("CSV: %s" % csv)
            if plait is not None:
                self._log_write(
                    "Ponto critico estimado (extrapolado, aproximado): "
                    "x1=%.4f  x2=%.4f  x3=%.4f" % (plait[0], plait[1], plait[2]))
            self._embed_figure(fig)
            self.btn_calc.config(state="normal")
            self._on_model_change()

        def _embed_figure(self, fig):
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            if self._canvas is not None:
                self._canvas.get_tk_widget().destroy()
            if self._fig is not None:
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            self._fig = fig
            self._canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self._canvas.draw()
            self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            self.plot_frame.rowconfigure(0, weight=1)
            self.plot_frame.columnconfigure(0, weight=1)

    root = tk.Tk()
    root.geometry("1040x720")
    app = EllApp(root)
    if _test_hook is not None:
        _test_hook(root, app)
        return
    root.mainloop()


if __name__ == "__main__":
    main()