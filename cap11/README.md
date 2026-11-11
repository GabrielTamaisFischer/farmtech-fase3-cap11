# Capítulo 11 — parte do Gabriel

Este diretório contém a preparação compartilhada, os experimentos individuais
e o notebook integrado do Capítulo 11.

## Fonte dos dados

`data/Atividade_Cap11_produtos_agricolas.csv` é uma cópia byte a byte do CSV
oficial recebido em `Atividade_Cap11_produtos_agricolas.csv`. O original não é
alterado. O notebook carrega essa cópia por caminho relativo.

## Integração para os cinco modelos

`modelagem.py` define o contrato comum que os integrantes podem reutilizar:

- `FEATURES` e `TARGET` definem as sete entradas e a classe `label`;
- `separar_treino_teste` gera um split estratificado único (20%, `random_state=42`);
- `construir_pipeline` mantém `StandardScaler` dentro do pipeline, ajustando-o
  somente no treino e novamente em cada fold;
- `criar_validacao_cruzada` fornece cinco folds estratificados compartilhados;
- `criar_tabela_comparacao` registra as métricas reais disponíveis e lista os
  modelos pendentes; `plotar_comparacao` exige os cinco modelos e as quatro
  métricas antes de gerar um gráfico, sem completar resultados ausentes.

O notebook `GabrielTamais_RM574503_fase3_cap11.ipynb` consolida EDA, comparação
de três culturas e os cinco algoritmos exigidos: Árvore de Decisão, Regressão
Logística, Random Forest, SVM e Gradient Boosting. Usa o split e a estratégia
de validação cruzada comuns, grava outputs executados e reúne métricas, matrizes
de confusão, diagnóstico de overfitting e importâncias quando disponíveis.

O notebook individual `fabio_arvore_decisao.ipynb` e o experimento
`gabriel_random_forest.ipynb` permanecem como artefatos de integração e
auditoria. Para reproduzir a validação completa, consulte `GUIA_TESTE.md`.

## Execução

Na raiz do repositório, instale `requirements.txt` e execute o notebook final:

```powershell
python -m nbconvert --to notebook --execute --inplace cap11/GabrielTamais_RM574503_fase3_cap11.ipynb
```

O notebook final não remove linhas, não imputa valores e não altera o dataset.
