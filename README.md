# B3Cam

Reconhecimento de gestos em tempo real que sobrepõe imagens acima do rosto do usuário usando a webcam.

## Funcionalidades

- Detecta 9 gestos diferentes em tempo real usando MediaPipe
- Detecta o rosto do usuário e posiciona a imagem do macaco acima da cabeça
- Suporte a duas mãos simultâneas para gestos especiais
- Suporte a câmera virtual (OBS, Zoom, etc.) via pyvirtualcam
- Transparência alpha em PNG para overlay suave

## Gestos Suportados

| Gesto | Dedos | Imagem |
|---|---|---|
| Jerry | Polegar levantado | `jerry.png` |
| Fazo L | Polegar + indicador (1 mão) | `fazoL.png` |
| Nerd | Indicador levantado | `nerd.png` |
| Time Shak | 4 dedos (sem polegar) | `time_shak.png` |
| Angry Ouvindo | Mão aberta (5 dedos) | `angry_ouvindo.png` |
| Son | Mão fechada (0 dedos) | `son.png` |
| Absolute Cinema | 2 mãos abertas | `absolute_cinema.png` |
| Go Drinking | 2 mãos com polegar+indicador | `go_drinking.png` |

## Requisitos

- Python 3.8+
- Webcam

## Instalação

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Uso

```bash
python gesture_tracker.py
```

Pressione `q` para sair.

## Estrutura do Projeto

```
B3cam/
├── gesture_tracker.py      # Aplicação principal
├── requirements.txt         # Dependências
├── models/                  # Modelos MediaPipe
│   ├── hand_landmarker.task
│   └── face_detector.tflite
├── monkey_images/           # Imagens dos gestos (PNG com transparência)
│   ├── absolute_cinema.png
│   ├── angry_ouvindo.png
│   ├── fazoL.png
│   ├── go_drinking.png
│   ├── jerry.png
│   ├── nerd.png
│   ├── son.png
│   └── time_shak.png
└── docs/
    └── README.md
```

## Câmera Virtual

Para transmitir para OBS/Zoom, instale:

```bash
pip install pyvirtualcam
```

A aplicação detecta automaticamente a disponibilidade e envia os frames.

## Creditos

peguei o programa do repositorio https://github.com/lapllacce/monkey-cam 
