"""
utils/scan_cameras.py
=====================
Utilitário para descobrir quais índices de câmera o OpenCV consegue abrir.

Uso:
    python -m utils.scan_cameras

Para cada índice testado, abre a câmera e mostra uma prévia de alguns frames
(com o texto do índice desenhado) em uma janela OpenCV. Use as setas/drag para
fechar, ou apenas feche a janela para avançar. Pressione uma tecla para seguir.

Dica: cubra UMA das câmeras por vez e observe qual janela mostra o índice.
"""
import cv2


def find_available_cameras(max_index: int = 6) -> list[int]:
    """Retorna os índices que conseguiram abrir uma câmera."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # tenta ler 1 frame para confirmar que há sinal
            ok, _ = cap.read()
            available.append((i, ok))
        cap.release()
    return available


def preview(index: int, duration_sec: int = 5) -> None:
    """Mostra a prévia de uma câmera com o índice escrito na tela."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[{index}] não abriu")
        return

    print(f"[{index}] mostrando prévia por {duration_sec}s — feche a janela para sair")
    import time
    end = time.time() + duration_sec
    while time.time() < end:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Camera index: {index}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(f"Preview cam {index}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Procurando câmeras disponíveis...")
    found = find_available_cameras()
    print("Índices que abriram:", found if found else "nenhuma encontrada")
    print()
    print("Agora vou mostrar uma prévia de cada câmera encontrada.")
    print("Cubra UMA câmera por vez e veja qual índice fica preto/escuro.")
    for idx, has_signal in found:
        if has_signal:
            preview(idx, duration_sec=6)
        else:
            print(f"[{idx}] abriu mas não capturou sinal (pode estar em uso)")
