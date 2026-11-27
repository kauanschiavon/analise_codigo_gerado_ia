1. Verificar o código gerado
Execute:
cat gerados/qwen2.5-coder-7b-automatico/HumanEval_000/sample_1.py


Confira se o arquivo contém a implementação completa da função.
2. Verificar a sintaxe
Execute:
python3 -m py_compile gerados/qwen2.5-coder-7b-automatico/HumanEval_000/sample_1.py


Se não aparecer nenhuma mensagem de erro, a sintaxe está válida.
3. Conferir os registros do experimento
Abra:
cat gerados/qwen2.5-coder-7b-automatico/manifest.jsonl


- prompt.txt: enunciado original.
- sample_N_completion.txt: resposta original do Qwen.
- sample_N.py: código extraído.
- config.json: configurações do experimento.
- manifest.jsonl: registro das gerações.
- errors.jsonl: registro de erros.
O __pycache__ é normal e pode ser ignorado.


O modelo conseguiu produzir 100 implementações.
Todas as 100 passaram pela validação sintática após a correção do extrator.
90 implementações passaram nos testes funcionais.
10 implementações apresentaram problemas de comportamento.
Nenhuma amostra ultrapassou o limite de tempo.


Problema	Aprovados	Reprovados
HumanEval/3	      1 	4
HumanEval/10	  0	    5
HumanEval/15	  4	    1

colocar oq significa a temperatura(tabela)(no pdf)

LLM Local — Experimento HumanEval
Scripts utilizados no experimento de geração, validação e análise de código produzido pelo Qwen 2.5 Coder 7B.
Arquivos
Arquivo	Para que serve
app.py	Arquivo principal da aplicação.
gerar_humaneval_corrigido.py	Envia os problemas do HumanEval ao Qwen pelo Ollama e salva as respostas geradas.
reconstruir_amostras.py	Reconstrói os arquivos .py a partir das respostas originais, preservando a indentação e removendo cercas Markdown.
validar.py	Faz uma verificação sintática preliminar dos arquivos Python com ast.parse().
validar_compile.py	Verifica se os arquivos realmente compilam com compile(), identificando erros como return fora de função.
testar.py	Executa os testes oficiais do HumanEval para cada amostra, usando Docker para isolar a execução.
analisar.py	Executa Pylint e Bandit nos códigos gerados e nas soluções humanas, produzindo relatórios de alertas.
.gitignore	Define quais arquivos e pastas o Git deve ignorar, como ambientes virtuais e caches.
README.md	Documenta resumidamente os arquivos do projeto.


Ordem geral de uso
1. gerar_humaneval_corrigido.py — gerar as respostas do modelo.
2. reconstruir_amostras.py — reconstruir os arquivos, quando necessário.
3. validar.py e validar_compile.py — verificar sintaxe e compilação.
4. testar.py — avaliar o comportamento com os testes do HumanEval.
5. analisar.py — executar a análise estática com Pylint e Bandit.
Execute os comandos a partir da pasta raiz do projeto. Consulte as opções de cada script com python3 nome_do_script.py --help, quando disponíveis.