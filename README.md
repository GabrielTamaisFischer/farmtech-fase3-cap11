# FIAP Fase 3 — Capítulo 11: Classificação de culturas

Projeto independente de análise exploratória e classificação de culturas agrícolas a partir do dataset acadêmico fornecido. Contém a preparação compartilhada, os experimentos individuais e o notebook integrado que compara os cinco modelos solicitados.

## Conteúdo

- `cap11/data/Atividade_Cap11_produtos_agricolas.csv`: cópia local do dataset oficial usado pelos notebooks.
- `cap11/modelagem.py`: contrato de features/target, split estratificado reprodutível, pipelines com `StandardScaler`, validação cruzada e comparação de métricas.
- `cap11/GabrielTamais_RM574503_fase3_cap11.ipynb`: notebook integrado com EDA, comparação de três culturas, cinco classificadores, métricas, matrizes de confusão e diagnóstico.
- `cap11/gabriel_random_forest.ipynb` e `cap11/fabio_arvore_decisao.ipynb`: experimentos individuais preservados para integração e auditoria.
- `cap11/GUIA_TESTE.md`: instruções detalhadas.
- `tests/test_cap11.py`: testes do contrato de dados, split, pipelines e comparações.

Os notebooks preservam os outputs já gerados, incluindo tabelas e imagens incorporadas. As execuções de validação devem gravar outputs em cópias temporárias, sem substituir os artefatos entregues.

## Instalação

Requer Python 3.10 ou superior. Na raiz deste repositório:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Em macOS/Linux, crie o ambiente com `python3 -m venv .venv` e ative com `source .venv/bin/activate`.

## Testes e verificações

```powershell
python -m unittest discover -s tests -p "test_cap11.py" -v
python -m compileall -q cap11 tests
python -m pip check
```

Para executar o notebook integrado, a partir da raiz:

```powershell
python -m nbconvert --to notebook --execute --inplace cap11/GabrielTamais_RM574503_fase3_cap11.ipynb
```

Os experimentos individuais podem ser reexecutados com:

```powershell
python -m nbconvert --to notebook --execute --inplace cap11/gabriel_random_forest.ipynb
python -m nbconvert --to notebook --execute --inplace cap11/fabio_arvore_decisao.ipynb
```

Os comandos `--inplace` atualizam os notebooks. Para manter os arquivos entregues intactos, faça uma cópia antes de reexecutá-los. Os três notebooks localizam `cap11/modelagem.py` e `cap11/data/` a partir da raiz ou da pasta `cap11`.

## Método compartilhado

O módulo `cap11/modelagem.py` define sete features (`N`, `P`, `K`, temperatura, umidade, pH e chuva) e o alvo `label`. O split estratificado usa 80/20 e `random_state=42`; a validação cruzada usa cinco folds. O scaler permanece dentro do pipeline para evitar ajuste prévio nos dados de teste. A comparação final só plota métricas efetivamente calculadas para os cinco modelos.

## Repositório GitHub planejado

Nome sugerido para publicação, ainda não criado: `GabrielTamaisFischer/farmtech-fase3-cap11`.
