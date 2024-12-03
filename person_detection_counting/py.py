import cv2
from ultralytics import YOLO
import numpy as np

# video dosyasını açıyoruz
kamera = cv2.VideoCapture(r"person.mp4")

# video dosyası açılmassa hata alıyoruz
if not kamera.isOpened():
    raise IOError("Video dosyası okunurken hata oluştu.")

# videonun özelliklerini alıyoruz yükseklik, genişlik, fps
genislik = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
yukseklik = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(kamera.get(cv2.CAP_PROP_FPS))

# çıktı videosu için videowriter oluşturduk
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_path = r"C:\Users\Emre\OneDrive\Masaüstü\image_processing_project\object_counting_output.avi"
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (genislik, yukseklik))

# yolo modelini alıyoruz
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print("Model yüklenirken hata oluştu:", e)
    exit()

# güven skorumuz
confidence_threshold = 0.5
# takip edilen nesnelerin merkez noktaları ve id leri tutar
izlenen_insanlar = {}
# toplam farklı insan sayısı
toplam_insan = 0
# nesne id si
yeni_insan_id = 0



while kamera.isOpened():
    success, frame = kamera.read()
    if not success:
        break

    results = model.predict(frame, conf=confidence_threshold)

    # framedeki tespit edilen kişilerin merkez noktaları
    gecerli_nesne = []

    for result in results:
        for r in result.boxes.data:
            class_id = int(r[5])
            if model.names[class_id] == "person":
                x1, y1, x2, y2 = map(int, r[:4])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                gecerli_nesne.append((cx, cy))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)



    # eşleşen nesneleri bul ve id ata
    for new_object in gecerli_nesne:
        min_distance = float('inf')
        closest_object_id = None

        for object_id, izlenen_insan in izlenen_insanlar.items():
            distance = np.sqrt((new_object[0] - izlenen_insan[0])**2 + (new_object[1] - izlenen_insan[1])**2)
            if distance < min_distance and distance < 50: # eşik değeri
                min_distance = distance
                closest_object_id = object_id
        
        if closest_object_id is None: # yeni insan
            izlenen_insanlar[yeni_insan_id] = new_object
            yeni_insan_id += 1
            toplam_insan += 1
        else:  
            izlenen_insanlar[closest_object_id] = new_object

        


    # takip edilen insanları çiz ve id lerini yazdır
    for object_id, (cx, cy) in izlenen_insanlar.items():
        cv2.putText(frame, str(object_id), (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Anlık ve toplam insan sayısını yazdır
    cv2.putText(frame, f'Cercevedeki insanlar: {len(gecerli_nesne)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Toplam insan: {toplam_insan}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



    video_writer.write(frame)

    cv2.imshow("nesne tanıma", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



kamera.release()
video_writer.release()
cv2.destroyAllWindows()

print(f'Toplam tespit edilen farklı insan sayısı: {toplam_insan}')