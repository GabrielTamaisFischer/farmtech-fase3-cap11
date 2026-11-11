"""Valida o contrato de preparação e a integração do Capítulo 11."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAP11_ROOT = PROJECT_ROOT / "cap11"
sys.path.insert(0, str(CAP11_ROOT))

from modelagem import (  # noqa: E402
    EXPECTED_COLUMNS,
    EXPECTED_MODELS,
    FEATURES,
    TARGET,
    construir_pipeline,
    carregar_dataset,
    criar_validacao_cruzada,
    criar_tabela_comparacao,
    plotar_comparacao,
    separar_treino_teste,
)


class TestCapitulo11(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dados = carregar_dataset(
            CAP11_ROOT / "data" / "Atividade_Cap11_produtos_agricolas.csv"
        )

    def test_contrato_e_fonte_oficial(self) -> None:
        self.assertEqual(self.dados.columns.tolist(), EXPECTED_COLUMNS)
        self.assertEqual(self.dados.shape, (2200, 8))
        self.assertEqual(self.dados[TARGET].nunique(), 22)
        self.assertFalse(self.dados.isna().any().any())
        self.assertEqual(int(self.dados.duplicated().sum()), 0)

    def test_split_reproduzivel_estratificado_e_sem_vazamento(self) -> None:
        primeiro = separar_treino_teste(self.dados)
        segundo = separar_treino_teste(self.dados)
        self.assertEqual(len(primeiro.X_train), 1760)
        self.assertEqual(len(primeiro.X_test), 440)
        self.assertEqual(list(primeiro.X_train.columns), FEATURES)
        self.assertNotIn(TARGET, primeiro.X_train.columns)
        self.assertFalse(set(primeiro.X_train.index) & set(primeiro.X_test.index))
        self.assertTrue(primeiro.X_train.equals(segundo.X_train))
        self.assertTrue(primeiro.y_test.equals(segundo.y_test))
        self.assertEqual(
            primeiro.y_test.value_counts().sort_index().to_dict(),
            {classe: 20 for classe in sorted(self.dados[TARGET].unique())},
        )

    def test_preprocessamento_esta_dentro_do_pipeline(self) -> None:
        from sklearn.preprocessing import StandardScaler

        from sklearn.ensemble import RandomForestClassifier

        pipeline = construir_pipeline(RandomForestClassifier(random_state=42))
        self.assertIsInstance(pipeline.named_steps["escala"], StandardScaler)
        self.assertEqual(pipeline.named_steps["modelo"].random_state, 42)

    def test_comparacao_exige_resultados_reais_dos_cinco_modelos(self) -> None:
        self.assertEqual(len(EXPECTED_MODELS), 5)
        disponiveis = criar_tabela_comparacao({"Random Forest": {"accuracy": None}})
        self.assertEqual(len(disponiveis), 1)
        self.assertEqual(
            disponiveis.attrs["modelos_pendentes"],
            [modelo for modelo in EXPECTED_MODELS if modelo != "Random Forest"],
        )
        with self.assertRaisesRegex(ValueError, "cinco modelos"):
            plotar_comparacao(disponiveis)

    def test_arvore_decisao_reutiliza_contrato_e_comparacao(self) -> None:
        particoes = separar_treino_teste(self.dados, random_state=42)
        self.assertEqual(list(particoes.X_train.columns), FEATURES)
        self.assertEqual(particoes.X_test.index.tolist(), separar_treino_teste(self.dados).X_test.index.tolist())
        self.assertFalse(set(particoes.X_train.index) & set(particoes.X_test.index))

        modelo = construir_pipeline(DecisionTreeClassifier(random_state=42))
        modelo.fit(particoes.X_train, particoes.y_train)
        previsoes = modelo.predict(particoes.X_test)
        self.assertEqual(len(previsoes), len(particoes.y_test))

        metricas = {
            "accuracy": accuracy_score(particoes.y_test, previsoes),
            "precision_macro": precision_score(particoes.y_test, previsoes, average="macro", zero_division=0),
            "recall_macro": recall_score(particoes.y_test, previsoes, average="macro", zero_division=0),
            "f1_macro": f1_score(particoes.y_test, previsoes, average="macro", zero_division=0),
        }
        self.assertTrue(all(np.isfinite(v) and 0 <= v <= 1 for v in metricas.values()))
        matriz = confusion_matrix(particoes.y_test, previsoes, labels=sorted(self.dados[TARGET].unique()))
        self.assertEqual(int(matriz.sum()), len(particoes.y_test))
        self.assertEqual(criar_validacao_cruzada().n_splits, 5)

        comparacao = criar_tabela_comparacao({
            "Árvore de Decisão": metricas,
            "Random Forest": {"accuracy": .9955, "precision_macro": .9957, "recall_macro": .9955, "f1_macro": .9955},
        })
        self.assertEqual(set(comparacao.index), {"Árvore de Decisão", "Random Forest"})
        self.assertEqual(comparacao.attrs["modelos_pendentes"], ["Regressão Logística", "SVM", "Gradient Boosting"])


if __name__ == "__main__":
    unittest.main()
