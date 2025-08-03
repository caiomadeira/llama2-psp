"""

"""

# nmap mapeia um arq. na memoria. Trata o arquivo como um grande
# array de bytes na RAM.
import mmap 
import struct # ponto entre converter dados do tipo python pro tipo C
import os
import argparse

"""
Extrai os parâmetros da arquitetura do modelo e criar o config.bin

dimension: a dimensão do vetor de cada token
hidden_dimension: a dimensão da camada interna na rede feed-forward
number_of_layers:  o número de blocos Transformer empilhados
number_of_heads: o numero de "attention heads" no mecanismo de atenção caracteristico da arq. transformer
number_key_value_heads: o número de heads pra key/value (importante pra otimizações
como grouped-query attention)
vocab_size: o número de tokens únicos do dicionario.
sequence_len: o comprimento máximo da sequência de tokens que o modelo pode processar
"""

class Config:
    def __init__(self):
        self.dimension = 0
        self.hidden_dimension = 0
        self.number_of_layers = 0
        self.number_of_heads = 0
        self.number_key_value_heads = 0
        self.vocab_size = 0
        self.sequence_len = 0

    # le a config do modelo e criar o arquivo config.bin
    def read_checkpoint(self, checkpoint, output_filename="config.bin"):
        with open(checkpoint, "rb") as file:
            """
            Calcula o tamanho de 7 inteiros 'iiiiiii'. Cada i representa
            um inteiro de 4 bytes. Total: 28 bytes.
            """
            config_data = file.read(struct.calcsize('iiiiiii')) # le os primeiros 28 bytes do stories260k.bin
            """
            o 'struct.unpack('iiiiiii')' interpreta os 28 bytes lidos como 7 inteiros
            de 4 bytes e os retorna como uma tupla de números python, que são atribuídos
            aos atributos da classe. 
            """
            (self.dimension, self.hidden_dimension, self.number_of_layers, self.number_of_heads,
             self.number_key_value_heads, self.vocab_size, self.sequence_len) = struct.unpack('iiiiiii', config_data)
            
            """
            isso é uma conveção do llama.c original. se o vocab_size no arquivo
            for negativo, é porque certos pesos não são compartilhados.
            capturamos essa info na variavel booleana 'shared_weights'
            e depois pegamos o valor absoluto pra yer o tamanho real do vocab.
            """
            shared_weights: bool = self.vocab_size > 0
            self.vocab_size = abs(self.vocab_size)

            # Abre o config.bin p/ escrita binária.
            """
            'h', self.dim - é a otimização principal. Pega o valor de self.dim
            e o "empacota" como um binário de 2 bytes. ('h' significa
            SHORT SIGNED INTEGER). Isso é feito pra todos os parametros.
            No meuc aso do PSP eu posso definir uma astruct nessa ordem e com tipos short
            ou int16_t e usar um único fread pra carregar o arquivo config.bin inteiro p/
            dentro da struct na memória do PSP.
            """
            with open(output_filename, "wb") as file:
                file.write(struct.pack('h', self.dimension))
                file.write(struct.pack('h', self.hidden_dimension))
                file.write(struct.pack('h', self.number_of_layers))
                file.write(struct.pack('h', self.number_of_heads))
                file.write(struct.pack('h', self.number_key_value_heads))
                file.write(struct.pack('h', self.vocab_size))
                file.write(struct.pack('h', self.sequence_len))
                file.write(struct.pack('h', int(shared_weights)))