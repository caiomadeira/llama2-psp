### nnet.cpp
#### rmsnorm

```
// o = x * w
void rmsnorm(float* o, float* x, float* weight, uint8_t size) {
    // calculando a soma dos quadrados
    float ss = 0.0f;
    for (uint16_t j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }   

    ss /= size;
    ss += 1e-5;
    ss = 1.0f / sqrtf(ss);

    // normaliza e escala
    for(uint16_t j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}
```

RMSNorm significa Root Mean Square Normalization (Normalização por Raiz da Média QUadrática)

Em uma rede neural, os numeros (ativactions) inputados passam de uma
camada pra outra e podem ficar muito grandes ou muito pequenos, o que causa instabilidade no treinamento e na inferência. A normalização serve pra "resetar" a escala desses números pra um nivel padrão e consistente.

__uint8__ para __uint16__:
Um uint8_t é um inteiro de 8 bits sem sinal. Seu valor máx é de 255.
Ou seja a função rmsnorm do nnet64 so consegue operar em vetores com no max
255 elementos. Para o PSP, se usamos uint16_t temos vetores com no max. 65.535 elementos. No caso do uint8_t pro c64 é uma limitação perigosa.