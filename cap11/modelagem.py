"""Preparação compartilhada e utilitários do Capítulo 11.

Todos os integrantes devem reutilizar o mesmo split e o mesmo pipeline para que
as métricas dos cinco algoritmos sejam comparáveis. O scaler fica dentro do
Pipeline, portanto é ajustado somente nos dados de treino de cada fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
EXPECTED_COLUMNS = FEATURES + [TARGET]
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
EXPECTED_MODELS = [
    "Árvore de Decisão",
    "Regressão Logística",
    "Random Forest",
    "SVM",
    "Gradient Boosting",
]


@dataclass(frozen=True)
class SplitDados:
    """Partições compartilhadas de treino e teste, sem transformação prévia."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def carregar_dataset(caminho: str | Path) -> pd.DataFrame:
    """Carrega o CSV e valida seu contrato sem modificar o arquivo de origem."""
    caminho = Path(caminho)
    dados = pd.read_csv(caminho)

    # O CSV oficial usa ``ph``; ``pH`` também é aceito conforme o enunciado.
    if "pH" in dados.columns and "ph" not in dados.columns:
        dados = dados.rename(columns={"pH": "ph"})

    if dados.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            "Schema inesperado. Esperado, nesta ordem: "
            f"{EXPECTED_COLUMNS}; recebido: {dados.columns.tolist()}"
        )
    if dados.empty:
        raise ValueError("O dataset oficial está vazio.")
    if dados[TARGET].isna().any():
        raise ValueError("A variável-alvo label contém valores ausentes.")
    return dados


def separar_treino_teste(
    dados: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> SplitDados:
    """Cria o split estratificado compartilhado, sem ajustar transformadores."""
    if TARGET in FEATURES or set(dados.columns) != set(EXPECTED_COLUMNS):
        raise ValueError("As colunas recebidas não correspondem ao contrato X/y.")

    X = dados.loc[:, FEATURES].copy()
    y = dados.loc[:, TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    if set(X_train.index).intersection(X_test.index):
        raise AssertionError("Os índices de treino e teste se sobrepõem.")
    return SplitDados(X_train, X_test, y_train, y_test)


def construir_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Aplica o pré-processamento comum dentro do pipeline do estimador."""
    return Pipeline(
        steps=[
            ("escala", StandardScaler()),
            ("modelo", estimator),
        ]
    )


def criar_validacao_cruzada(
    *, n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE
) -> StratifiedKFold:
    """Retorna o splitter estratificado a ser reutilizado pelos cinco modelos."""
    return StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )


def criar_tabela_comparacao(resultados: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Organiza métricas reais disponíveis e informa nomes de modelos pendentes."""
    if not resultados:
        raise ValueError("Nenhuma métrica de modelo foi fornecida.")
    tabela = pd.DataFrame.from_dict(resultados, orient="index")
    tabela.index.name = "Modelo"
    ausentes = [nome for nome in EXPECTED_MODELS if nome not in tabela.index]
    tabela.attrs["modelos_pendentes"] = ausentes
    return tabela


def plotar_comparacao(tabela: pd.DataFrame) -> plt.Figure:
    """Plota as cinco métricas de modelo somente após a integração completa."""
    modelos_ausentes = [nome for nome in EXPECTED_MODELS if nome not in tabela.index]
    if modelos_ausentes:
        raise ValueError(
            "A comparação visual exige resultados reais dos cinco modelos. "
            f"Ainda faltam: {modelos_ausentes}"
        )
    colunas = [coluna for coluna in METRICS if coluna in tabela.columns]
    metricas_ausentes = [metrica for metrica in METRICS if metrica not in colunas]
    if metricas_ausentes:
        raise ValueError(f"Faltam métricas obrigatórias: {metricas_ausentes}")
    if tabela[colunas].isna().any().any() or not tabela[colunas].apply(
        lambda serie: serie.between(0, 1).all()
    ).all():
        raise ValueError("As métricas devem ser numéricas, finitas e estar entre 0 e 1.")

    figura, eixo = plt.subplots(figsize=(10, 5.5))
    tabela[colunas].plot(kind="bar", ax=eixo, ylim=(0, 1), width=0.78)
    eixo.set_title("Comparação dos modelos — métricas disponíveis")
    eixo.set_xlabel("Algoritmo")
    eixo.set_ylabel("Métrica (0 a 1)")
    eixo.grid(axis="y", alpha=0.25)
    eixo.legend(title="Métrica", loc="best")
    eixo.tick_params(axis="x", rotation=15)
    figura.tight_layout()
    return figura


def configurar_estilo_graficos() -> None:
    """Define estilo visual consistente para os gráficos do Capítulo 11."""
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.prop_cycle": plt.cycler(color=["#0072B2", "#D55E00", "#009E73"]),
            "figure.figsize": (10, 6),
        }
    )
