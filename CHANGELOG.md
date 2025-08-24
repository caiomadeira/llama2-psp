# CHANGELOG

### 24/06/25 - Solving some problems and official release

You can get the release in release page.


In generate.cpp, the function generate, is trying to process and print all the tokens, which includes the input
token prompts. Btw, the first token of a sequence is usually a beginning of sentence (BOS) control token. This token
doesn't habe a normal char and must appears a "memory trash".

```
if (next == 1) then
```
Was give me some problems. The token with ID 1 is the BOS (Beginning of Sentence) token, but can be used as EOS (End of Sentence) token too. This condition guarantees that the model generate the text and insert the "end token" by your criteria. This means a good quality answer (usually), but can be shorter. If i remove this verification, the model will ignore the "end token" and will continue "talking" while is above the limit of __steps__. This means repeated tokens and non sense answers.

### 09/08/25
Successfully executed on physical PSP hardware, but with some crash problems. 

### 03/08/25
Working only on PPSSPP. Using [Tiny Stories 260k model](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K).  

![Running on PPSSPP 2](assets/1.png)
