### Generating model files

O conjunto de scripts Python pega o modelo Llama2 treinado e o converte para arquivos binários simples e otimizados que o hardware limitado entende.

__stories260k.bin__(model file): Esse é o arquivo do modelo (não do Llama2 completo da Meta). 
É um modelo muito pequeno, baseado na arquitetura do Llama2, que foi treinado com o dataset "TinyStories". Tem 260 mil parâmetros.

__tok512.bin__(tokenizador): Uma espécie de "dicionário" que corresponde ao arquivo do modelo.
O arquivo converte palabras em números (tokens) - e vice-versa.

O conjunto de scripts usa eles como input e deve retornar: __config.bin__ com a arquitetura do modelo, __tokenizer.bin__ o dicionario de tokens formatado p/ codigo C e uma variação do __weights.reu__, o arquivo mais importante, contento todos os pesos da rede neural prontos pra serem carregados.