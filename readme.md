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

colocar oq significa a temperatura(tabela)