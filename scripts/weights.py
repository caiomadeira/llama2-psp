import mmap 
import struct
import os
import argparse

"""
Extrai os pesos numericos do modelo e cria o arquivo de pesos
weights_data: armazena os dados dos pesos
"""
class Weights:
    def __init__(self):
        self.weights_data = None
    # o parametro 'checkpoint' da função se refere a 'model checkpoint' um termo de ML p/ o arquivo que salva o estado de um modelo treinado.
    def read_weights(self, checkpoint, output_filename="weights.psp"):
        with open(checkpoint, "rb") as file:
           file.seek(28) # pula o primeiros 28 bytes (Config)
           self.weights_data = file.read()
        
        """
        Escreve uma espécie de "assinatura"(magic number) de 4 bytes.
        Isso permite que seu programa em C verifique se o arquivo é valido
        antes de carregar. L264, acredito que signifique Llama2 para C64.
        No meu caso, mudei para L2PS.

        Eu so posso armazenar a string 'L2PS' Pois o tipo de dado uint32_t
        so pode armazenar 4 carcrtres (4 bytes).
        """
        with open(output_filename, "wb") as file:
           file.write('L2PS'.encode('utf-8'))
           file.write(self.weights_data)

        self.pad_to_next_multiple(output_filename)

    """
    Essa função adiciona bytes nulos ao final do arquivo pro tamanho total
    do mesmo ser um múltiplo exato de 2mb, 4mb, etc. No contexto do REU
    do Commodore 64 é uma otimização necessária. No caso do PSP talvez
    mude um pouco pela presença de 32mb.
    """
    def pad_to_next_multiple(self, filename, multiples=(2, 4, 8, 16)):
        file_size = os.path.getsize(filename=filename)
        """
        a 'next_multiple' encontra o menor valor da lista 'multiples'
        (em bytes) que seja mior que o tamanho atual do arquivo
        """
        next_multiple = min(m for m in multiples if m * 1024 * 1024 > file_size)
        print("file size: ", file_size)
        print("next multiple:", next_multiple)
        padding_size = next_multiple * 1024 * 1024 - file_size
        print("padding size (remainer bytes to reach this multiple size): ", padding_size)

        # apenas faz a anexação binária (ab) p/ adicionar os dados ao final sem apagar os dados que ja existem
        with open(filename, "ab") as file:
            file.write(b'\0' * padding_size)