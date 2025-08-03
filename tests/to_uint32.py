import struct

def calcular_assinatura(texto_de_4_chars):
    if len(texto_de_4_chars) != 4:
        return "Erro: A string precisa ter exatamente 4 caracteres."
    bytes_da_string = texto_de_4_chars.encode('ascii')
    valor_inteiro = struct.unpack('<I', bytes_da_string)[0]
    return f'0x{valor_inteiro:08X}'

assinatura_original = calcular_assinatura('L264')
print(f"A assinatura para 'L264' é: {assinatura_original}")
assinatura_nova = calcular_assinatura('L2PS')
print(f"A assinatura para 'L2PS' é: {assinatura_nova}")
assinatura_invalida = calcular_assinatura('L2PSP')
print(f"Tentativa com 'L2PSP': {assinatura_invalida}")