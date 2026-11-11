# FIAP Fase 3 — Capítulo 11: classificação de culturas

## Resumo do trabalho

Este projeto explora o dataset acadêmico `cap11/data/Atividade_Cap11_produtos_agricolas.csv` e compara classificadores para identificar a cultura indicada por sete atributos de solo e clima (`N`, `P`, `K`, temperatura, umidade, pH e chuva). A base contém 2.200 registros, 22 culturas e 100 registros por cultura. O notebook final reúne EDA, perfil de referência, avaliação controlada de cinco modelos e conclusões. Os resultados medem capacidade de classificar este CSV e não demonstram causalidade, produtividade ou recomendação de plantio.

## Grupo 27

| Integrante | RM |
|---|---:|
| Bruna Collavite | 576285 |
| Gabriel Tamais Fischer | 574503 |
| Fabio Guntovitch | 575582 |
| Natasha Zonta | 574230 |
| Fabrizzio Riccelli | 576083 |

Os nomes e RMs acima são os registrados nos materiais do projeto integrado. O notebook final identifica Gabriel Tamais Fischer como autor do relatório consolidado.

## Dataset e metodologia

O dataset incluído é a cópia usada no trabalho; o notebook verifica o schema e executa a análise sem sobrescrever o CSV. Cada linha tem sete atributos numéricos e o rótulo `label`. As classes estão balanceadas: cada uma das 22 culturas tem 100 linhas. A partição é estratificada, fixa `random_state=42` e separa 1.760 amostras de treino e 440 de teste. `StandardScaler` fica dentro do pipeline e é ajustado somente no treino de cada fold, evitando vazamento de informação. A validação cruzada estratificada usa cinco folds dentro do treino; o teste é reservado para avaliação final.

## Análise exploratória e interpretação dos gráficos

Os gráficos desta seção estão incorporados aos outputs do [notebook final](cap11/GabrielTamais_RM574503_fase3_cap11.ipynb); não há cópias PNG independentes no repositório.

### Distribuição das culturas e variáveis

O gráfico de contagem mostra barras iguais, com 100 observações em cada classe. Assim, não há prevalência de uma cultura que, por si só, domine a acurácia global; a avaliação macro ainda é apresentada para atribuir peso igual a cada cultura. Os histogramas de `N`, `P`, `K`, temperatura, umidade, pH e chuva mostram escalas e distribuições diferentes. Isso motiva a padronização dentro dos pipelines dos modelos sensíveis a escala, em vez de transformar dados antes da divisão treino/teste.

### Correlações entre atributos

O mapa de correlação usa correlação linear de Pearson entre as sete variáveis numéricas. A maior associação absoluta é entre fósforo e potássio (r ≈ 0,736); N e P têm correlação negativa menor (r ≈ −0,231), e temperatura/umidade apresentam associação positiva fraca (r ≈ 0,205). O padrão indica que algumas variáveis variam juntas nesta base e que relações simples podem ajudar a distinguir classes. Correlação não implica causalidade nem demonstra que alterar um nutriente produzirá determinada cultura.

### Perfil de referência e comparação de três culturas

O “perfil ideal” pedido no exercício é tratado com cautela: como não há produtividade, rendimento ou indicador de sucesso agronômico no CSV, o projeto define a **mediana global das condições rotuladas** como perfil ideal de referência empírico. É um resumo estatístico do centro do dataset, não um ótimo agronômico nem recomendação universal.

| Atributo | Mediana global de referência | `rice` | `maize` | `chickpea` |
|---|---:|---:|---:|---:|
| N | 37 | 80 | 76 | 39 |
| P | 51 | 47 | 48,5 | 68 |
| K | 32 | 40 | 20 | 79 |
| Temperatura | 25,599 | 23,735 | 22,844 | 18,878 |
| Umidade | 80,473 | 82,189 | 65,304 | 16,659 |
| pH | 6,425 | 6,364 | 6,259 | 7,357 |
| Chuva | 94,868 | 233,120 | 83,472 | 79,692 |

O heatmap padronizado compara as medianas de cada cultura com essa referência. `rice` se destaca principalmente por chuva muito acima da mediana global, além de N e K mais altos; sua umidade também está ligeiramente acima. `maize` tem N alto, umidade e K abaixo da referência, enquanto temperatura e pH ficam relativamente próximos. `chickpea` apresenta umidade muito abaixo, K e P acima, temperatura abaixo e pH acima. Os gráficos de distribuição por cultura reforçam que os perfis não são um único vetor “ideal” compartilhado: cada rótulo está associado a uma combinação diferente. Essas diferenças descrevem como a base rotulada foi construída e não indicam que ajustar o solo para essas medianas aumentaria rendimento.

## Modelos e avaliação

Os cinco algoritmos usam a mesma partição treino/teste, o mesmo esquema de validação cruzada e pipelines comparáveis. Precision, Recall e F1 são médias macro (média das classes com peso igual). A tabela reproduz os outputs medidos no notebook final; métricas no teste e F1 da CV não devem ser comparadas como se fossem uma única amostra.

| Modelo | Accuracy teste | Precision macro | Recall macro | F1 macro teste | F1 macro médio na CV (DP) |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0,9955 | 0,9957 | 0,9955 | 0,9955 | 0,9931 (0,0059) |
| Gradient Boosting | 0,9886 | 0,9897 | 0,9886 | 0,9887 | 0,9869 (0,0040) |
| SVM (RBF) | 0,9841 | 0,9856 | 0,9841 | 0,9840 | 0,9788 (0,0119) |
| Árvore de Decisão | 0,9795 | 0,9806 | 0,9795 | 0,9794 | 0,9852 (0,0076) |
| Regressão Logística | 0,9727 | 0,9740 | 0,9727 | 0,9725 | 0,9678 (0,0076) |

O gráfico comparativo resume as quatro métricas no teste. O Random Forest teve o maior F1 macro no teste (0,9955) e maior F1 médio na CV (0,9931); o gap F1 treino-teste foi 0,0045. Ele também teve o melhor valor de cada métrica na tabela do teste. Isso sustenta sua escolha como melhor resultado **neste experimento**: o ensemble combina árvores e pode capturar fronteiras não lineares sem o gap maior da árvore única. Não prova superioridade em dados novos. O Gradient Boosting alcançou o segundo F1 no teste e a menor variação entre folds (DP 0,0040), embora seu gap treino-teste (0,0113) seja maior. A Árvore de Decisão teve F1 médio de CV (0,9852) superior ao próprio teste (0,9794), com gap 0,0206, sinal para examinar estabilidade e possível ajuste excessivo. O SVM teve variação de CV maior (DP 0,0119), e a Regressão Logística obteve o menor F1 nos resultados registrados, compatível com fronteiras lineares menos flexíveis para padrões complexos.

As cinco matrizes de confusão do notebook colocam classe real nas linhas e prevista nas colunas. A maioria dos registros está na diagonal; entre os erros listados, `rice` previsto como `jute` aparece em quatro casos na Regressão Logística, três no SVM, dois no Gradient Boosting e um no Random Forest. Os erros são poucos e concentrados em pares específicos, mas isso pode refletir sobreposição de atributos nesta amostra. Importâncias por impureza são apresentadas apenas para árvores e ensembles; os coeficientes da Regressão Logística são uma medida diferente. Não se atribui importância artificial comparável ao SVM RBF.

## Conclusões, pontos fortes e limitações

O dataset balanceado permitiu comparar cinco classificadores com uma divisão reproduzível, CV no treino, preprocessing encapsulado para evitar vazamento e avaliação macro. O Random Forest teve melhor resultado neste conjunto de teste e desempenho consistente na validação cruzada. Os perfis de `rice`, `maize` e `chickpea` mostram combinações diferentes de condições registradas, e não um padrão agrícola único.

**Pontos fortes:** dados locais reproduzíveis, contrato explícito de atributos e alvo, split estratificado compartilhado, pipelines, métricas macro, matrizes de confusão e comparação dos cinco modelos com resultados calculados.

**Limitações:** não há produtividade ou sucesso agronômico como alvo; os rótulos não permitem inferir causalidade; o dataset pode não representar outras regiões, safras, equipamentos ou distribuições futuras; não há validação externa; o split é uma única partição; importâncias por impureza podem favorecer atributos com mais pontos de corte; e o melhor resultado pode mudar com novos dados ou ajuste de hiperparâmetros. Antes de qualquer uso operacional, é necessária validação independente e revisão agronômica.

## Como reproduzir

Requer Python 3.10 ou superior. A partir da raiz:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m compileall -q cap11 tests
python -m pip check
```

O **notebook final de entrega à FIAP** é [`cap11/GabrielTamais_RM574503_fase3_cap11.ipynb`](cap11/GabrielTamais_RM574503_fase3_cap11.ipynb). Para reproduzir seus outputs em uma cópia (sem alterar o artefato versionado), execute:

```powershell
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 --output-dir "$env:TEMP\cap11-validado" --output notebook_validado.ipynb cap11/GabrielTamais_RM574503_fase3_cap11.ipynb
```

Os notebooks individuais `cap11/fabio_arvore_decisao.ipynb` e `cap11/gabriel_random_forest.ipynb` permanecem para integração e auditoria; as tabelas comparativas finais são as do notebook consolidado. Instruções adicionais estão em [`cap11/GUIA_TESTE.md`](cap11/GUIA_TESTE.md).

## Estrutura

```text
farmtech-fase3-cap11/
├── README.md
├── requirements.txt
├── cap11/
│   ├── data/              # dataset usado na análise
│   ├── modelagem.py       # contrato de dados, split e pipelines
│   ├── *.ipynb            # relatório final e notebooks individuais
│   └── GUIA_TESTE.md
└── tests/                 # testes do contrato e modelagem
```

Repositório: [GabrielTamaisFischer/farmtech-fase3-cap11](https://github.com/GabrielTamaisFischer/farmtech-fase3-cap11).
