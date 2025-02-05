# Hackathon IA para devs

## Desafio
A FIAP VisionGuard, empresa de monitoramento de câmeras de segurança, está analisando a viabilidade de uma nova funcionalidade para otimizar o seu software.
O objetivo da empresa é usar de novas tecnologias para identificar situações atípicas e que possam colocar em risco a segurança de estabelecimentos e comércios que utilizam suas câmeras.
Um dos principais desafios da empresa é utilizar Inteligência Artificial para identificar objetos cortantes (facas, tesouras e similares) e emitir alertas para a central de segurança.
A empresa tem o objetivo de validar a viabilidade dessa feature, e para isso será necessário fazer um MVP para detecção supervisionada desses objetos.

## Objetivos
- Construir ou buscar um dataset contendo imagens de facas, tesouras e outros objetos cortantes em diferentes condições de ângulo e iluminação.
- Anotar o dataset para treinar o modelo supervisionado, incluindo imagens negativas (sem objetos perigosos) para reduzir falsos positivos.
- Treinar o modelo
- Desenvolver um sistema de alertas (pode ser e-mail)

## Solução
Primeiramente buscamos um dataset contendo diversas armas brancas para que pudessemos fazer o treinamento do nosso modelo. O dataset escolhido foi [weapon-detection-knifes](https://universe.roboflow.com/morcik-fix/weapon-detection-knifes-mohip-kmkw6).

Antes de executar o treinamento e o teste, é necessário executar
```
pip install ultralytics
```

Para realizar o treinamento localmente é necessário fazer o download do dataset e executar o arquivo 
```
python train.py
````

Após finalizado o treinamento é possível testar o modelo e as notificações de detecção executando o arquivo 
```
python test.py
```

## Métricas do treinamento
```
Precisão (Precision): 0.67
Recall: 0.41
mAP@50: 0.67
mAP@50-95: 0.39
```

