# -*- coding: utf-8 -*-
"""
Adaptador COSMO-SAC via JCOSMO (gateway py4j). Expoe a MESMA assinatura
`gamma(x)` da classe UnifacLL, de modo que motor, estabilidade, flash e
diagrama permanecam agnosticos ao modelo.

API py4j confirmada a partir dos scripts de exemplo LVPP/UFRGS (Ferrarini
et al. 2018) ja validados neste projeto:

    from py4j.java_gateway import JavaGateway
    gateway = JavaGateway(auto_field=True)
    JCOSMO = gateway.entry_point
    JCOSMO.listModels()                        # lista modelos disponiveis
    model = JCOSMO.newModel('COSMO-SAC-HB2 (GAMESS)')
    comps = gateway.new_array(gateway.jvm.java.lang.String, n)
    comps[0] = 'TRIOLEIN'; comps[1] = 'ETHANOL'; ...
    model.setCompounds(comps)
    model.setTemperature(298.15)
    x = gateway.new_array(gateway.jvm.double, n)
    x[0] = 0.5; ...
    model.setComposition(x)
    lnGamma = model.activityCoefficientLn()    # retorna ln(gamma), nao gamma

Requisitos de execucao (LOCAIS, nao disponiveis neste ambiente):
  - JCOSMO aberto (GUI) atuando como servidor py4j ANTES do calculo.
  - py4j instalado (pip install py4j; testado com py4j 0.9).
  - Perfis sigma dos PILs gerados como PAR IONICO NEUTRO via GAMESS
    (nao separar cation e anion; a separacao produz gamma ~ 10^9-10^16).
  - Java 21: JCOSMO deve ser iniciado com a flag de modulo
        --add-opens java.base/java.lang=ALL-UNNAMED
    para contornar o erro de reflexao do XStream (sistema de modulos).

Modelo no JCOSMO: 'COSMO-SAC-HB2 (GAMESS)'.
"""

import re

import numpy as np

JCOSMO_MODEL_DEFAULT = "COSMO-SAC-HB2 (GAMESS)"
X_FLOOR = 1e-8           # piso de composicao antes de chamar o JCOSMO
CACHE_DECIMALS = 9       # arredondamento da chave de cache


class CosmoSacError(RuntimeError):
    """Erro de conexao ou de calculo do lado do JCOSMO."""
    pass


def _parse_model_names(raw):
    """listModels() retorna uma UNICA string Java com os nomes entre aspas
    simples, cada um seguido da citacao DOI, separados por '\\n'
    (nao um array). Extrai os nomes de dentro das aspas."""
    s = str(raw)
    names = re.findall(r"'([^']*)'", s)
    if names:
        return names
    # formato inesperado: tenta tratar como iteravel de nomes ja prontos
    try:
        return [str(v).strip() for v in raw]
    except TypeError:
        return [s.strip()]


class CosmoSacAdapter:

    def __init__(self, jcosmo_names, T=298.15,
                 model_name=JCOSMO_MODEL_DEFAULT, gateway=None):
        """
        jcosmo_names : lista com os 3 nomes EXATOS dos compostos no banco do
                       JCOSMO, na ordem (oleo/acido, alcool, PIL).
        gateway      : JavaGateway ja aberto (opcional). Se None, abre uma
                       nova conexao com auto_field=True (igual aos scripts
                       de exemplo LVPP/UFRGS).
        """
        if len(jcosmo_names) != 3:
            raise ValueError("sao necessarios exatamente 3 nomes de composto")
        self.n = 3
        self.T = float(T)
        self.names = list(jcosmo_names)
        self.model_name = model_name
        self._cache = {}
        self._calls = 0
        self._model = None

        self._connect(gateway)
        self._setup_model()

    # ---- conexao ao gateway py4j ----
    def _connect(self, gateway):
        if gateway is not None:
            self.gateway = gateway
            return
        try:
            from py4j.java_gateway import JavaGateway
        except ImportError:
            raise CosmoSacError(
                "py4j nao esta instalado. Instale com 'pip install py4j' "
                "e abra o JCOSMO antes de rodar o COSMO-SAC.")
        try:
            self.gateway = JavaGateway(auto_field=True)
            self.jcosmo_entry = self.gateway.entry_point
            _ = self.jcosmo_entry.listModels()   # toque de sanidade
        except Exception as exc:
            raise CosmoSacError(
                "JCOSMO deve estar aberto antes de rodar calculos COSMO-SAC "
                "(gateway py4j inacessivel). Detalhe: %s" % exc)

    # ---- selecao de modelo e configuracao dos compostos ----
    def _setup_model(self):
        """Cria o modelo via newModel() e registra os tres compostos de
        uma vez via setCompounds() com array Java de String."""
        try:
            raw_models = self.jcosmo_entry.listModels()
            available = _parse_model_names(raw_models)
            if self.model_name.strip() not in available:
                raise CosmoSacError(
                    "Modelo '%s' nao encontrado no JCOSMO. Modelos "
                    "disponiveis: %s" % (self.model_name, available))
            self._model = self.jcosmo_entry.newModel(self.model_name)

            comps = self.gateway.new_array(
                self.gateway.jvm.java.lang.String, self.n)
            for i, nome in enumerate(self.names):
                comps[i] = nome
            self._model.setCompounds(comps)
            self._model.setTemperature(float(self.T))
        except CosmoSacError:
            raise
        except Exception as exc:
            raise CosmoSacError(
                "Falha ao configurar o modelo/compostos no JCOSMO. "
                "Verifique se os nomes batem exatamente com o banco. "
                "Detalhe: %s" % exc)

    # ---- nucleo do calculo: ln gamma para uma composicao ja tratada ----
    def _compute_ln_gamma(self, x):
        """Retorna o vetor ln gamma do JCOSMO para a composicao x (ja com
        piso e renormalizada). x e lista de floats nativos do Python.

        IMPORTANTE: a composicao vai como array Java de double
        (gateway.new_array), nao como lista Python, e cada valor passa por
        float() nativo antes da atribuicao — numpy 2.0+ representa
        np.float64 de forma que gera NumberFormatException no lado Java.
        """
        xj = self.gateway.new_array(self.gateway.jvm.double, self.n)
        for i, v in enumerate(x):
            xj[i] = float(v)
        self._model.setComposition(xj)
        ln_g = self._model.activityCoefficientLn()
        return np.array([float(v) for v in ln_g], dtype=float)

    # ---- interface publica (mesma assinatura de UnifacLL.gamma) ----
    def gamma(self, x):
        x = np.asarray(x, dtype=float)
        x = np.clip(x, X_FLOOR, None)     # piso para evitar NaN no solver interno
        x = x / x.sum()                   # renormaliza apos o clip

        key = tuple(np.round(x, CACHE_DECIMALS))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        x_native = [float(v) for v in x]  # floats nativos para o py4j
        try:
            ln_g = self._compute_ln_gamma(x_native)
        except CosmoSacError:
            raise
        except Exception as exc:
            raise CosmoSacError("erro no calculo COSMO-SAC via JCOSMO: %s" % exc)

        g = np.exp(ln_g)
        self._calls += 1
        self._cache[key] = g
        return g

    @property
    def n_calls(self):
        return self._calls