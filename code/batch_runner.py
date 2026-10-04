# -*- coding: utf-8 -*-
"""Executor de fila (batch) para varreduras de ELL.

Le uma lista de sistemas de um arquivo CSV e roda cada um em sequencia,
salvando o diagrama (PNG) e o CSV de tie-lines de cada, alem de um resumo
consolidado ao final. Projetado para rodar sem supervisao: um sistema que
falha (ex.: JCOSMO fora do ar, composto sem perfil sigma, parametro
ausente) e REGISTRADO e a fila SEGUE para o proximo, em vez de abortar tudo.

Formato do CSV de entrada (cabecalho obrigatorio; ver fila_exemplo.csv):

    oleo,alcool,li,modelo,n_grid
    trioleina,etanol,2HEAHX,UNIFAC-LL,20
    trioleina,etanol,2HEAHX,COSMO-SAC,20
    ...

- 'oleo', 'alcool', 'li'  -> chaves internas (as mesmas do dropdown; ver
  OIL_OPTIONS/ALCOHOL_OPTIONS/IL_OPTIONS em interface_ell.py).
- 'modelo'                -> "UNIFAC-LL" ou "COSMO-SAC".
- 'n_grid'                -> opcional; se vazio, usa o padrao (20).

Uso por linha de comando:
    python batch_runner.py fila.csv
    python batch_runner.py fila.csv --outdir resultados_fila

Uso programatico (a interface chama assim):
    from batch_runner import run_batch
    run_batch("fila.csv", outdir="resultados_fila", log_cb=print,
              progress_cb=lambda i, n, nome: ...)
"""

import os
import csv
import time
import subprocess
import traceback

from interface_ell import (compute_equilibrium, render_and_export,
                            COMPONENT_LABEL, JCOSMO_DEFAULTS)


DEFAULT_N_GRID = 20


class _InibirHibernacao:
    """Gerenciador de contexto que impede o computador de hibernar/suspender
    enquanto a fila roda (Linux com systemd). Usa 'systemd-inhibit' segurando
    um bloqueio de sleep+idle num processo filho que dorme ate ser encerrado;
    ao sair do contexto, o processo e morto e o bloqueio liberado
    automaticamente. Se 'systemd-inhibit' nao existir (sistema sem systemd),
    apenas avisa e segue sem inibir — nunca trava a fila por causa disso."""

    def __init__(self, log=None, why="rodando fila de ELL"):
        self.log = log or (lambda s: None)
        self.why = why
        self.proc = None

    def __enter__(self):
        try:
            # mantem um processo vivo (sleep infinito) sob o inibidor; o
            # bloqueio dura enquanto esse processo existir.
            self.proc = subprocess.Popen(
                ["systemd-inhibit",
                 "--what=sleep:idle",
                 "--who=Diagramas ELL",
                 "--why=%s" % self.why,
                 "--mode=block",
                 "sleep", "infinity"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.log("Hibernacao inibida durante a fila (systemd-inhibit ativo).")
        except FileNotFoundError:
            self.proc = None
            self.log(">> Aviso: 'systemd-inhibit' nao encontrado; a fila vai "
                     "rodar, mas o computador pode hibernar. Para evitar, "
                     "ajuste as configuracoes de energia manualmente.")
        except Exception as e:
            self.proc = None
            self.log(">> Aviso: nao foi possivel inibir hibernacao (%s). "
                     "A fila segue normalmente." % e)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.log("Bloqueio de hibernacao liberado.")
        return False   # nao suprime excecoes


def _resolver_jcosmo_names(comp_keys):
    """Nomes padrao no banco do JCOSMO para as chaves dadas (usado apenas
    pelo COSMO-SAC)."""
    return [JCOSMO_DEFAULTS.get(k, k.upper()) for k in comp_keys]


def ler_fila(csv_path):
    """Le o CSV de fila e devolve uma lista de dicts, um por sistema.
    Ignora linhas em branco e linhas iniciadas por '#'. Valida colunas
    minimas; levanta ValueError com mensagem clara se o cabecalho faltar."""
    sistemas = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        # tolera separador ';' ou ',' (Excel BR costuma salvar com ';')
        amostra = f.read(2048)
        f.seek(0)
        sep = ";" if amostra.count(";") > amostra.count(",") else ","
        reader = csv.DictReader(f, delimiter=sep)
        campos = [c.strip().lower() for c in (reader.fieldnames or [])]
        obrig = {"oleo", "alcool", "li", "modelo"}
        if not obrig.issubset(set(campos)):
            raise ValueError(
                "CSV precisa ter as colunas: oleo, alcool, li, modelo "
                "(n_grid e opcional). Encontrado: %s" % campos)
        for i, row in enumerate(reader, start=2):
            row = {(k or "").strip().lower(): (v or "").strip()
                   for k, v in row.items()}
            if not row.get("oleo") or row.get("oleo", "").startswith("#"):
                continue
            try:
                n_grid = int(row["n_grid"]) if row.get("n_grid") else DEFAULT_N_GRID
            except ValueError:
                n_grid = DEFAULT_N_GRID
            sistemas.append({
                "linha": i,
                "oleo": row["oleo"],
                "alcool": row["alcool"],
                "li": row["li"],
                "modelo": row["modelo"] or "UNIFAC-LL",
                "n_grid": n_grid,
                "T": float(row["t"]) if row.get("t") else 298.15,
            })
    return sistemas


def run_batch(csv_path, outdir="resultados_fila", T_default=298.15,
              log_cb=None, progress_cb=None, stop_flag=None):
    """Roda a fila inteira. Robusto a falhas: cada sistema roda dentro de
    try/except; erro em um nao interrompe os demais.

    log_cb(str)             -> recebe mensagens de progresso (default: print)
    progress_cb(i, n, nome) -> chamado ao iniciar cada sistema (opcional)
    stop_flag()             -> se fornecido e retornar True, a fila para
                               graciosamente apos o sistema corrente
    Retorna a lista de resultados (um dict por sistema, com status).
    """
    log = log_cb or (lambda s: print(s))
    os.makedirs(outdir, exist_ok=True)

    sistemas = ler_fila(csv_path)
    n = len(sistemas)
    log("Fila carregada: %d sistema(s) a processar." % n)
    resultados = []
    t_inicio = time.time()

    with _InibirHibernacao(log=log):
        for idx, s in enumerate(sistemas, start=1):
            if stop_flag is not None and stop_flag():
                log(">> Fila interrompida pelo usuario apos %d de %d." % (idx - 1, n))
                break

            comp_keys = [s["oleo"], s["alcool"], s["li"]]
            try:
                labels = [COMPONENT_LABEL.get(k, k) for k in comp_keys]
            except Exception:
                labels = comp_keys
            nome = " + ".join(labels) + "  [%s]" % s["modelo"]

            if progress_cb is not None:
                progress_cb(idx, n, nome)
            log("\n[%d/%d] %s (n_grid=%d, T=%.2f)"
                % (idx, n, nome, s["n_grid"], s["T"]))

            registro = {"nome": nome, "modelo": s["modelo"], "linha": s["linha"],
                        "status": None, "tie_lines": None, "png": None,
                        "erro": None, "tempo": None}
            t0 = time.time()
            try:
                jcosmo_names = None
                if s["modelo"] == "COSMO-SAC":
                    jcosmo_names = _resolver_jcosmo_names(comp_keys)

                out = compute_equilibrium(
                    comp_keys, s["modelo"], s["T"], s["n_grid"],
                    jcosmo_names=jcosmo_names, log_cb=log)

                png, csv_path_out, plait = render_and_export(
                    out, labels, outdir=outdir)
                registro["status"] = "ok"
                registro["tie_lines"] = out.get("n_accepted", 0)
                registro["png"] = png
                registro["unreliable"] = out.get("unreliable", False)
                log("   -> OK: %d tie-lines | %s"
                    % (out.get("n_accepted", 0), os.path.basename(png)))

            except Exception as e:
                registro["status"] = "erro"
                registro["erro"] = "%s: %s" % (type(e).__name__, e)
                log("   -> ERRO (registrado, seguindo para o proximo): %s"
                    % registro["erro"])
                for linha in traceback.format_exc().splitlines():
                    log("      " + linha)

            registro["tempo"] = time.time() - t0
            resultados.append(registro)

    # resumo consolidado
    _escrever_resumo(resultados, outdir, time.time() - t_inicio, log)
    return resultados


def _escrever_resumo(resultados, outdir, tempo_total, log):
    """Escreve um CSV-resumo de tudo que rodou (sucesso e falha)."""
    resumo_path = os.path.join(outdir, "resumo_fila.csv")
    with open(resumo_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sistema", "modelo", "status", "tie_lines",
                    "confiavel", "tempo_s", "png", "erro"])
        for r in resultados:
            w.writerow([
                r["nome"], r["modelo"], r["status"],
                r["tie_lines"] if r["tie_lines"] is not None else "",
                "nao" if r.get("unreliable") else "sim",
                "%.1f" % r["tempo"] if r["tempo"] else "",
                os.path.basename(r["png"]) if r["png"] else "",
                r["erro"] or "",
            ])

    n_ok = sum(1 for r in resultados if r["status"] == "ok")
    n_erro = sum(1 for r in resultados if r["status"] == "erro")
    n_bin = sum(1 for r in resultados if r["status"] == "ok" and (r["tie_lines"] or 0) > 0)
    log("\n" + "=" * 60)
    log("FILA CONCLUIDA em %.1f s" % tempo_total)
    log("  sucesso: %d | erro: %d | com binodal: %d" % (n_ok, n_erro, n_bin))
    log("  resumo salvo em: %s" % resumo_path)
    log("=" * 60)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Executor de fila de ELL")
    ap.add_argument("csv", help="arquivo CSV com a fila de sistemas")
    ap.add_argument("--outdir", default="resultados_fila",
                    help="pasta de saida (default: resultados_fila)")
    args = ap.parse_args()
    run_batch(args.csv, outdir=args.outdir)