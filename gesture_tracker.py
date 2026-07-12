#!/usr/bin/env python3
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import math
# ============================================================
# IMPORTAÇÕES
# ============================================================
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles, HandLandmarksConnections, FaceDetector, FaceDetectorOptions
from pygrabber.dshow_graph import FilterGraph
import numpy as np


# ============================================================
# CLASSE PRINCIPAL: GestureTracker
# ============================================================
class GestureTracker:
    """
    Classe responsável pelo rastreamento e reconhecimento de gestos.
    
    Atributos:
        mp_hands: Módulo de detecção de mãos do MediaPipe
        mp_drawing: Utilitários de desenho do MediaPipe
        mp_drawing_styles: Estilos de desenho do MediaPipe
        hands: Instância do detector de mãos
        monkey_images: Dicionário com as imagens dos gestos do macaco
        current_gesture: Gesto atualmente detectado
    """
    
    def __init__(self):
        """Inicializa o rastreador de gestos e carrega os recursos necessários."""
        
        # ========================================
        # Inicialização do MediaPipe (Tasks API)
        # ========================================
        model_path = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
        
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
        
        # ========================================
        # Inicialização do Detector de Rosto (Tasks API)
        # ========================================
        face_model_path = os.path.join(os.path.dirname(__file__), "models", "face_detector.tflite")
        
        face_base_options = mp_python.BaseOptions(model_asset_path=face_model_path)
        face_options = mp_vision.FaceDetectorOptions(
            base_options=face_base_options,
            min_detection_confidence=0.5
        )
        self.face_detector = mp_vision.FaceDetector.create_from_options(face_options)
        
        # ========================================
        # Recursos do Sistema
        # ========================================
        self.monkey_images = self.load_monkey_images()  # Carrega imagens dos gestos
        self.current_gesture = "neutral"                 # Gesto inicial: neutro
        
    # ========================================
    # MÉTODO: Carregar Imagens
    # ========================================
    def load_monkey_images(self):
        """
        Carrega as imagens dos gestos do macaco da pasta 'monkey_images/'.
        
        Returns:
            dict: Dicionário com {nome_do_gesto: imagem_carregada}
        """
        images = {}
        image_dir = "monkey_images"
        
        # Verificar se o diretório existe
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            print(f"📁 Criado diretório {image_dir}")
            print("⚠️  Adicione imagens de macacos nesta pasta:")
            print("   - neutral.png (posicao neutra)")
            print("   - absolute_cinema.png (duas maos abertas)")
            print("   - nerd.png (dedo indicador levantado)")
            print("   - jerry.png (dedao levantado)")
            print("   - fazoL.png (polegar + indicador, 1 mao)")
            print("   - son.png (mao fechada)")
        
        # Mapeamento dos gestos e seus arquivos
        gesture_files = {
            "neutral": "neutral.png",
            "absolute_cinema": "absolute_cinema.png",
            "nerd": "nerd.png",
            "jerry": "jerry.png",
            "fazoL": "fazoL.png",
            "angry_ouvindo": "angry_ouvindo.png",
            "time_shak": "time_shak.png",
            "son": "son.png",
            "go_drinking": "go_drinking.png"
        }
        
        # Carregar cada imagem
        for gesture, filename in gesture_files.items():
            filepath = os.path.join(image_dir, filename)
            
            if os.path.exists(filepath):
                # Ler imagem (IMREAD_UNCHANGED preserva canal alpha/transparência)
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                
                if img is not None:
                    # Redimensionar para tamanho padrão (300x300 pixels)
                    img = cv2.resize(img, (300, 300))
                    images[gesture] = img
                    print(f"✅ Carregada: {filename}")
                else:
                    print(f"❌ Erro ao ler: {filename}")
            else:
                print(f"⚠️  Não encontrada: {filename}")
        
        return images
    
    # ========================================
    # MÉTODO: Contar Dedos Levantados
    # ========================================
    def count_fingers(self, hand_landmarks, handedness):
        """
        Conta quantos dedos estão levantados com base nos landmarks da mão.
        
        Args:
            hand_landmarks: Lista de landmarks (21 pontos) da mão
            handedness: "Right" ou "Left" (não usado, mantido por compatibilidade)
        
        Returns:
            list: Lista de 5 elementos [polegar, indicador, médio, anelar, mínimo]
                  onde 1 = levantado e 0 = abaixado
        """
        fingers_up = []
        
        # IDs dos landmarks importantes
        finger_tips = [4, 8, 12, 16, 20]    # Pontas dos 5 dedos
        finger_pips = [3, 6, 10, 14, 18]    # Articulações para comparação
        
        # ========================================
        # POLEGAR (lógica por distância)
        # ========================================
        # Compara distância da ponta (4) vs articulação (3) até a base do indicador (5)
        # Quando o polegar está estendido, a ponta fica mais longe da base do indicador
        tip = hand_landmarks[4]
        ip = hand_landmarks[3]
        base = hand_landmarks[5]
        
        dist_tip = math.hypot(tip.x - base.x, tip.y - base.y)
        dist_ip = math.hypot(ip.x - base.x, ip.y - base.y)
        
        if dist_tip > dist_ip:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
        
        # ========================================
        # OUTROS DEDOS (lógica vertical)
        # ========================================
        for i in range(1, 5):
            if hand_landmarks[finger_tips[i]].y < hand_landmarks[finger_pips[i]].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        return fingers_up
    
    # ========================================
    # MÉTODO: Detectar Gesto
    # ========================================
    def detect_gesture(self, hand_landmarks, handedness):
        fingers = self.count_fingers(hand_landmarks, handedness)
        fingers_count = sum(fingers)

        if fingers == [1, 0, 0, 0, 0]:
            return "jerry"

        if fingers == [1, 1, 0, 0, 0]:
            return "fazoL"

        if fingers == [0, 1, 0, 0, 0]:
            return "nerd"

        if fingers == [0, 1, 1, 1, 1]:
            return "time_shak"

        if fingers_count == 5:
            return "angry_ouvindo"

        if fingers_count == 0:
            return "son"

        return None
    
    # ========================================
    # MÉTODO: Sobrepor Imagem
    # ========================================
    def overlay_image(self, background, overlay, x, y):
        """
        Sobrepõe uma imagem (overlay) sobre outra (background) com suporte a transparência.
        
        Args:
            background: Imagem de fundo (frame da câmera)
            overlay: Imagem a ser sobreposta (imagem do macaco)
            x: Posição X (horizontal) onde colocar a imagem
            y: Posição Y (vertical) onde colocar a imagem
        
        Returns:
            numpy.ndarray: Imagem de fundo com overlay aplicado
        
        Nota:
            Suporta imagens PNG com canal alpha (transparência)
        """
        if overlay is None:
            return background
        
        h, w = overlay.shape[:2]  # Altura e largura do overlay
        
        # ========================================
        # Ajustar Tamanho se Não Couber na Tela
        # ========================================
        if x + w > background.shape[1]:
            w = background.shape[1] - x
            overlay = cv2.resize(overlay, (w, h))
        
        if y + h > background.shape[0]:
            h = background.shape[0] - y
            overlay = cv2.resize(overlay, (w, h))
        
        # Verificar se posição é válida
        if x < 0 or y < 0:
            return background
        
        # ========================================
        # Aplicar Transparência (Canal Alpha)
        # ========================================
        if overlay.shape[2] == 4:  # Imagem tem canal alpha (RGBA)
            # Normalizar alpha de 0-255 para 0-1
            alpha = overlay[:, :, 3] / 255.0
            
            # Misturar cada canal de cor (B, G, R)
            for c in range(3):
                background[y:y+h, x:x+w, c] = (
                    alpha * overlay[:, :, c] +                    # Parte visível do overlay
                    (1 - alpha) * background[y:y+h, x:x+w, c]    # Parte visível do fundo
                )
        else:  # Imagem sem transparência (RGB)
            background[y:y+h, x:x+w] = overlay
        
        return background
    
    # ========================================
    # MÉTODO: Listar Câmeras Disponíveis
    # ========================================
    def list_cameras(self):
        available_cameras = []
        seen = set()
        print("\n🔍 Procurando câmeras disponíveis...")

        graph = FilterGraph()
        try:
            dshow_names = graph.get_input_devices()
        except Exception:
            dshow_names = []

        backends = [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
        ]

        for i, dshow_name in enumerate(dshow_names):
            for backend_label, backend_id in backends:
                try:
                    cap = cv2.VideoCapture(i, backend_id)
                except Exception:
                    continue
                if cap.isOpened():
                    ret, _ = cap.read()
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    cap.release()
                    if ret and w > 0:
                        key = (i, w)
                        if key not in seen:
                            seen.add(key)
                            available_cameras.append({
                                "index": i,
                                "backend": backend_id,
                                "name": dshow_name
                            })

        for i in range(10):
            for backend_label, backend_id in backends:
                try:
                    cap = cv2.VideoCapture(i, backend_id)
                except Exception:
                    continue
                if cap.isOpened():
                    ret, _ = cap.read()
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    cap.release()
                    if ret and w > 0:
                        key = (i, w)
                        if key not in seen:
                            seen.add(key)
                            available_cameras.append({
                                "index": i,
                                "backend": backend_id,
                                "name": f"Câmera {i}"
                            })

        return available_cameras

    def select_camera(self):
        cameras = self.list_cameras()

        if not cameras:
            print("❌ Nenhuma câmera encontrada!")
            return None

        print(f"\n📹 Câmeras disponíveis:")
        for cam in cameras:
            print(f"   [{cam['index']}] {cam['name']}")

        if len(cameras) == 1:
            print(f"\n✅ Usando: {cameras[0]['name']}")
            return cameras[0]

        while True:
            try:
                options = [str(c["index"]) for c in cameras]
                choice = input(f"\nEscolha o índice da câmera ({options[0]} padrão): ").strip()

                if choice == "":
                    print(f"✅ Usando: {cameras[0]['name']}")
                    return cameras[0]

                choice_int = int(choice)
                valid = [c["index"] for c in cameras]
                if choice_int in valid:
                    chosen = next(c for c in cameras if c["index"] == choice_int)
                    print(f"✅ Usando: {chosen['name']}")
                    return chosen
                else:
                    print(f"⚠️  Câmera {choice_int} não disponível. Opções: {valid}")

            except ValueError:
                print("⚠️  Digite um número válido.")
            except KeyboardInterrupt:
                print("\n\n❌ Operação cancelada pelo usuário.")
                return None
    
    # ========================================
    # MÉTODO PRINCIPAL: Loop de Execução
    # ========================================
    def run(self, camera_id=None):
        if camera_id is None:
            camera_id = self.select_camera()
            if camera_id is None:
                return

        if isinstance(camera_id, dict):
            cam_index = camera_id["index"]
            cam_backend = camera_id["backend"]
            cam_name = camera_id["name"]
        else:
            cam_index = camera_id
            cam_backend = cv2.CAP_ANY
            cam_name = f"Câmera {camera_id}"

        cap = cv2.VideoCapture(cam_index, cam_backend)

        if not cap.isOpened():
            print(f"❌ Erro: Não foi possível abrir a câmera {cam_name}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width == 0 or height == 0:
            width, height = 640, 480

        print("\n" + "=" * 60)
        print("🎥 CÂMERA INICIADA!")
        print("=" * 60)
        print(f"📹 Usando câmera: {cam_name} ({width}x{height})")
        print("\n👋 GESTOS DISPONÍVEIS:")
        print("   🙌  Duas maos abertas = ABSOLUTE CINEMA")
        print("   🤙  Polegar + indicador (2 maos) = GO DRINKING")
        print("   ☝️  Dedo indicador levantado = Nerd")
        print("   👍  Dedao levantado = Jerry")
        print("   🤙  Polegar + indicador (1 mao) = Fazo L")
        print("   🖐️  5 dedos levantados = Angry")
        print("   🖐️  4 dedos levantados = Time Shak")
        print("   ✊  Mao fechada = Son")
        print("   😐  Neutro (sem gesto)")
        print("\n⌨️  Pressione 'q' para sair")
        print("=" * 60 + "\n")

        cv2.namedWindow('Gesture Tracker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gesture Tracker', width, height)
        cv2.moveWindow('Gesture Tracker', 50, 100)

        virtual_cam = None
        try:
            import pyvirtualcam
            virtual_cam = pyvirtualcam.Camera(width=width, height=height, fps=30)
            print(f"📺 pyvirtualcam ativo: {virtual_cam.device}")
        except Exception as e:
            print(f"⚠️  pyvirtualcam indisponível ({e})")

        gesture_names = {
            "neutral": "Neutro",
            "absolute_cinema": "ABSOLUTE CINEMA",
            "nerd": "Nerd",
            "jerry": "Jerry",
            "fazoL": "Fazo L",
            "angry_ouvindo": "Angry",
            "time_shak": "Time Shak",
            "son": "Son",
            "go_drinking": "Go Drinking"
        }

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            results = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            # ========================================
            # DETECÇÃO DE ROSTO
            # ========================================
            face_results = self.face_detector.detect(mp_image)
            face_bbox = None
            if face_results.detections:
                detection = face_results.detections[0]
                bbox = detection.bounding_box
                face_bbox = {
                    "x": bbox.origin_x,
                    "y": bbox.origin_y,
                    "width": bbox.width,
                    "height": bbox.height
                }

            if results.hand_landmarks and results.handedness:
                all_hands = []
                for hand_landmarks, handedness in zip(
                    results.hand_landmarks,
                    results.handedness
                ):
                    hand_label = handedness[0].category_name
                    all_hands.append((hand_landmarks, hand_label))

                self.current_gesture = "neutral"

                if len(all_hands) >= 2:
                    both_open = True
                    for hl, hl_label in all_hands:
                        fingers = self.count_fingers(hl, hl_label)
                        if sum(fingers) != 5:
                            both_open = False
                            break
                    if both_open:
                        self.current_gesture = "absolute_cinema"

                    if self.current_gesture == "neutral":
                        both_drink = True
                        for hl, hl_label in all_hands:
                            fingers = self.count_fingers(hl, hl_label)
                            if fingers != [1, 1, 0, 0, 0]:
                                both_drink = False
                                break
                        if both_drink:
                            self.current_gesture = "go_drinking"

                if self.current_gesture == "neutral":
                    for hl, hl_label in all_hands:
                        gesture = self.detect_gesture(hl, hl_label)
                        if gesture is not None:
                            self.current_gesture = gesture
                            break
            else:
                self.current_gesture = "neutral"

            gesture_text = gesture_names.get(self.current_gesture, self.current_gesture)

            if self.current_gesture in self.monkey_images and self.current_gesture != "neutral" and face_bbox is not None:
                monkey_img = self.monkey_images[self.current_gesture]
                
                # Calcular tamanho proporcional ao rosto (80% da largura do rosto)
                face_width = face_bbox["width"]
                thumb_size = int(face_width * 0.8)
                
                # Garantir tamanho mínimo e máximo
                thumb_size = max(80, min(thumb_size, 300))
                
                thumb = cv2.resize(monkey_img, (thumb_size, thumb_size))

                # Posicionar acima da cabeça, centralizado horizontalmente
                face_center_x = face_bbox["x"] + face_bbox["width"] // 2
                x0 = face_center_x - thumb_size // 2
                y0 = face_bbox["y"] - thumb_size  # Acima do topo do rosto

                # Ajustar limites da tela
                x0 = max(0, min(x0, width - thumb_size))
                y0 = max(0, min(y0, height - thumb_size))

                if thumb.shape[2] == 4:
                    alpha = thumb[:, :, 3:] / 255.0
                    roi = image[y0:y0 + thumb_size, x0:x0 + thumb_size]
                    image[y0:y0 + thumb_size, x0:x0 + thumb_size] = (
                        alpha * thumb[:, :, :3] + (1 - alpha) * roi
                    ).astype(np.uint8)
                else:
                    image[y0:y0 + thumb_size, x0:x0 + thumb_size] = thumb

            cv2.imshow('Gesture Tracker', image)

            if virtual_cam is not None:
                output_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                virtual_cam.send(output_rgb)
                virtual_cam.sleep_until_next_frame()

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("\n👋 Saindo...")
                break

        cap.release()
        cv2.destroyAllWindows()
        if virtual_cam is not None:
            virtual_cam.close()
        print("✅ Programa encerrado com sucesso!")
        print("=" * 60 + "\n")

# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def main():
    """
    Ponto de entrada do programa.
    
    Função:
        - Exibe banner inicial
        - Cria instância do rastreador
        - Inicia o loop de detecção
    """
    # Banner de boas-vindas
    print("\n" + "=" * 60)
    print("🐒 RASTREADOR DE GESTOS COM MACACO 🐒")
    print("=" * 60)
    print("Sistema de reconhecimento de gestos em tempo real")
    print("Desenvolvido com OpenCV e MediaPipe")
    print("=" * 60)
    
    try:
        # Criar e executar o rastreador
        tracker = GestureTracker()
        tracker.run()
    
    except KeyboardInterrupt:
        print("\n\n❌ Programa interrompido pelo usuário (Ctrl+C)")
        print("=" * 60 + "\n")
    
    except Exception as e:
        print(f"\n\n❌ Erro inesperado: {e}")
        print("=" * 60 + "\n")
        raise


# ============================================================
# EXECUÇÃO DO PROGRAMA
# ============================================================
if __name__ == "__main__":
    main()
