# Guia para executar e testar o Capítulo 11

O dataset usado pelos notebooks está em `cap11/data/` e é a cópia do CSV oficial.

## 0. Obter a versão mais recente

O repositório independente planejado é `farmtech-fase3-cap11`. Para clonar após sua publicação:

```powershell
git clone https://github.com/GabrielTamaisFischer/farmtech-fase3-cap11.git
cd farmtech-fase3-cap11
```

Para atualizar um clone já existente:

```powershell
git -C farmtech-fase3-cap11 pull origin main
cd farmtech-fase3-cap11
```

Execute os demais comandos a partir da raiz do repositório. O arquivo final a
entregar à FIAP é `cap11/GabrielTamais_RM574503_fase3_cap11.ipynb`.

## 1. Preparar o ambiente

No Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se `py` não estiver disponível, use `python -m venv .venv` no primeiro
comando. Em macOS/Linux, ative com `source .venv/bin/activate`.

## 2. Rodar a suíte de testes

```powershell
python -m unittest discover -s tests -v
```

Os testes verificam o schema e as dimensões do dataset, a estratificação e
reprodutibilidade do split, a ausência de sobreposição e o contrato dos
pipelines e da tabela comparativa.

## 3. Abrir e executar o notebook final

Abra `cap11/GabrielTamais_RM574503_fase3_cap11.ipynb` no Jupyter Notebook,
JupyterLab ou VS Code. Para reexecutá-lo integralmente e gravar os outputs:

```powershell
python -m nbconvert --to notebook --execute --inplace cap11/GabrielTamais_RM574503_fase3_cap11.ipynb
```

As tabelas, gráficos, métricas e conclusões ficam salvos no próprio notebook.

## 4. Executar os notebooks individuais

Para conferir os experimentos que alimentam a integração:

```powershell
python -m nbconvert --to notebook --execute --inplace cap11/gabriel_random_forest.ipynb
python -m nbconvert --to notebook --execute --inplace cap11/fabio_arvore_decisao.ipynb
```

Os três comandos usam o mesmo `modelagem.py`, as mesmas colunas, o split
estratificado com `random_state=42` e os mesmos cinco folds definidos no
projeto.

## 5. Conferências manuais no notebook final

- O dataset deve mostrar 2.200 linhas, 8 colunas e 22 culturas.
- O split deve mostrar 1.760 linhas de treino e 440 de teste, sem índices
  sobrepostos.
- A EDA contém mais de cinco gráficos e compara `rice`, `maize` e `chickpea`.
- A seção de CV apresenta média e desvio padrão de Accuracy, Precision macro,
  Recall macro e F1 macro para cada algoritmo.
- A avaliação final mostra as quatro métricas no mesmo teste, cinco matrizes de
  confusão, principais confusões, diagnóstico treino-teste e tabela comparativa.

O perfil de referência é a mediana global do dataset e serve apenas como base
descritiva de comparação. Não deve ser interpretado como recomendação agronômica
universal.
