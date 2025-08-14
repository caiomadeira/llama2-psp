# Llama2 PSP
Inference Llama 2 in C on Playstation Portable (PSP) BY Caio Madeira.  
Based on inference of ytmytm to [Llama 2 for c64](https://github.com/ytmytm/llama2.c64).

![Running on PPSSPP](assets/1.png)

## CHANGELOG
09/08/25 - Successfully executed on physical PSP hardware, but with some crash problems.
03/08/25 - Funcionando apenas no PPSSPP. Using [Tiny Stories 260k model](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K). 

## Portando funções
O ponto central da adaptação é: atribuir o acesso ao hardware da REU por acesso aos ponteiros da RAM principal do PSP.

### Tokenizer.cpp x tokenizer64.c
#### Arquitetura MIPS do PSP X x86 p/ PCs
A CPU MIPS do PSP é litte-endian. O código original do c64
que escreve os dados binários (short e float) também escreve
no formato little-endian (o byte menos significativo primeiro).

### Transformer.cpp x transformer64.c
A struct TransformerWeights64 usa typedef uint32_t REUPtr.
Um REUPtr é um endereço absoluto de 32 bits na memória REU do
commodore 64.   

No contexto do PSP, carrego o weights.psp inteiro em um único
bloco de memória alocado com malloc. Logo, todos os campos
do tipo REUPtr não serão endereços absolutos na memória
externa e sim ponteiros float que apontam pra diferentes
locais dentro desse grande bloco de memória.

Posso manter o tipo REUPtr mas mudando o seu significado pra ser um offset a partir do inicio do meu bloco de memoria de pesos ou apenas mudar tudo pro float*.

### nnet.cpp x nnet64.c
Esse arquivo é "cérebro" da inferência. Ele contém as implementações dos algoritmos do Transformer, como a multiplicação de matrizes, normalização e a attention. A maior parte do trabalho computacional pesado ocorre aqui.
No meu caso, removi as chamadas de REU_getf e REU_putf e substitui pelo acesso
direto a memória do PSP.

### math.c (da versão c64)
Não precisamos. Usaremos o <math.h>.

### generate.cpp x generatec64.c
O generate é o loop principal que produz o texto, token por token.
É um orquestrador do processo de geração de texto junto com o sampler64.c (ou sampler.cpp).

