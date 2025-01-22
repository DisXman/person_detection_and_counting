# person_detection_and_counting

Nesne Sayma ve Takip - YOLOv8 Kullanımı

Bu Python kodu, bir video üzerinde gerçek zamanlı nesne algılama ve takip işlemleri gerçekleştirir. Projenin ana hedefi, videodaki bireyleri ("person" sınıfı) algılayarak takip etmek ve toplam benzersiz kişi sayısını hesaplamaktır.


Özellikler

YOLOv8 Entegrasyonu: Yüksek performanslı nesne algılama için YOLOv8 modeli kullanılır.
Nesne Takibi: Algılanan kişileri kareler arasında takip eder ve benzersiz ID'ler atar.
Kişi Sayma: Video boyunca algılanan toplam benzersiz kişi sayısını hesaplar.
Video Çıktısı: İşlenmiş videoya, her karede kişi sınırlayıcı kutuları, ID'leri, anlık ve toplam kişi sayısı eklenir.

Nasıl Kullanılır?
İşlenecek video dosyasını (örn. person.mp4) uygun bir konuma yerleştirin.
Kodda video giriş ve çıkış yollarını güncelleyin.
YOLOv8 model dosyasının (yolov8n.pt) ortamınızda mevcut olduğundan emin olun.
Kodunuzu çalıştırarak videoyu işleyin ve açıklamalı sonucu elde edin.
