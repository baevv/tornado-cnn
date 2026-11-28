# Sunum Notları · Ana Sayfa

Bu dosya, arayüzdeki ana sayfa sunumunun konuşma metnidir. Slaytlarda **yalnızca başlık, etiket ve sayılar** yer alır; anlatım buradadır.

**Gezinme:** → / sağ tık = sonraki kare (bölümün son karesinden sonra sonraki slayt) · ← = geri · Boşluk = oynat/durdur

**Sayılar:** Slayttaki tüm sayılar modellerin `results.json`, `history.json` ve `preds.parquet` dosyalarından okunur. Bu notlarda sayı yazılmaz: "ekrandaki değer" dendiğinde ekrandan okunur. Model yeniden eğitilince notlar değişmez.

**Slayt yazı kuralı:** Bir karede en fazla bir başlık + birkaç etiket/sayı. Cümle yok.

**Temel 3 model:** Sunum ve arayüz (üst bardaki "Temel" görünümü) yalnız üç modeli anlatır: doğrulamada en iyi TornadoCNN varyantı, en iyi ResNet-18 ve öteki ResNet-18. Seçim her açılışta doğrulama PR-AUC'sine göre yapılır (seed'i olan deneylerde ortalama); test kümesi seçimde hiç kullanılmaz. Bugün: CNN · CoordConv, ResNet-18 · ön-eğitimli, ResNet-18 · sıfırdan. Yeniden eğitimden sonra kazanan değişirse slaytlar da değişir; notlardaki model adlarını ekrandan oku.

**Kronoloji:** Sunum projeyi yaptığımız sırayla anlatır. Bir sonuç, onu üreten adım anlatılmadan ekrana gelmez: kapakta ve veri bölümlerinde yalnız veri sayıları var, model skorları eğitimden sonra, test sonuçları ise yalnız 8. bölümde görünür.

---

## 1 · Kapak

**Ekranda:** Dönen radar taraması · "Radar Görüntülerinden Hortum Tespiti" · 2 sayı (örnek sayısı, hortum oranı). Statik slayt: hepsi birlikte gelir, sayılar sıfırdan sayar. Model sonucu yok.

**Konuşma:**
> Projemizde bir fırtınanın radar görüntüsüne bakarak o anda yerde hortum olup olmadığını tahmin eden evrişimli sinir ağları geliştirdik. Bu kadar örnekle çalıştık ve bunların yalnızca bu kadarı hortum. Bu iki sayı, işin neden zor olduğunu da özetliyor. Şimdi projeyi yaptığımız sırayla, veriden başlayarak anlatacağım.

---

## 2 · Veri seti (2 statik slayt)

### Slayt 2a · TorNet

**Ekranda:** Örneklerden oluşan ABD haritası (gri: örnek yoğunluğu, kırmızı: hortum, halka: radar) · örnek, yıl, radar sayıları · TOR / WRN / NUL şeridi. Harita batıdan doğuya açılır.

**Konuşma:**
> Veri olarak MIT Lincoln Laboratuvarı'nın açık veri seti TorNet'i kullandık. ABD'deki NEXRAD hava radarlarından ekrandaki yıllar arasında toplanmış bu kadar fırtına örneği var. Haritadaki her nokta örneklerin geldiği yer; harita çizilmedi, örneklerin konumları kendiliğinden ülkenin şeklini veriyor. Halkalar radar istasyonları, kırmızılar hortumların yoğunlaştığı yerler: ortadaki "hortum koridoru" belirgin.
>
> Etiketler üç gruptan geliyor. TOR doğrulanmış hortum; modelin bulmasını istediğimiz pozitif sınıf bu. WRN, uyarı verilmiş ama hortum çıkmamış fırtınalar; hortuma çok benzedikleri için en zor örnekler bunlar. NUL ise rastgele seçilmiş sıradan fırtınalar. Alttaki şerit, hortumun veride ne kadar az yer kapladığını gösteriyor.

### Slayt 2b · Kullandığımız veri

**Ekranda:** 100 noktalık ızgara (hortumlar en son, sağ altta) · "k / 100" · yıl çubukları (kullanılan yıllar koyu) · EF çubukları.

**Konuşma:**
> Bütün veri çok büyük olduğu için üç yıl seçtik; koyu çubuklar bunlar. Kırmızı alt kısımlar her yıldaki hortum örnekleri.
>
> Bu veriden rastgele 100 örnek alsak yalnızca bu kadarı hortum olurdu. Bu çok dengesiz bir veri: model her zaman "hortum yok" dese bile doğruluğu çok yüksek çıkar. Bu yüzden başarıyı doğrulukla değil, nadir sınıfı ne kadar iyi bulduğunu ölçen PR-AUC ile değerlendireceğiz.
>
> Alttaki çubuklar hortumların şiddeti. Çoğu zayıf (EF0–EF1); güçlü hortumlar az ama radarda çok daha belirgin, zayıflar ise modeli en çok zorlayacak olanlar.

---

## 3 · Radar görüntüsü (3 slayt)

Örnek fırtına otomatik seçilir: EF2 ve üstü hortum kareleri arasında merkezde dönmesi en güçlü olan (01 defteri C2 ile aynı ölçü). Yer, tarih ve istasyon ekrandan okunur. Renkler arayüzdeki Gezgin ile aynıdır: VEL'de yeşil radara yaklaşan, kırmızı uzaklaşan rüzgâr.

### Slayt 3a · Radar nasıl ölçer? (4 kare, geçişler animasyonlu)

**Kare 1 · Ekranda:** Radar, mesafe halkaları, dönen ışın · fırtına gerçek yerinde (DBZ)
> Radar anteni kendi etrafında döner, her yöne bir ışın gönderir ve geri dönen yansımayı uzaklığa göre ölçer. Burada gördüğünüz, seçtiğimiz fırtınanın radara göre gerçek konumu.

**Kare 2 · Ekranda:** Dilim vurgulanır · "60° × 60 km"
> Veri setindeki bir örnek radarın bütün çemberi değil: fırtınanın çevresinden kesilmiş 60 derecelik, 60 kilometrelik bir dilim. Modele verdiğimiz şey yalnızca bu dilim.

**Kare 3 · Ekranda:** Dilim döner ve dikdörtgene açılır · eksenler "açı · 120 × 0,5°", "mesafe · 240 × 250 m"
> Radar veriyi harita olarak değil tablo olarak kaydeder: her yön bir satır, her mesafe bir sütun. Dilimi açtığımızda 120'ye 240'lık bir dikdörtgen elde ediyoruz. Model fırtınayı bu şekilde, bükülmemiş haliyle görüyor.

**Kare 4 · Ekranda:** Radara en yakın ve en uzak sütunda iki kutu · gerçek genişlikleri (m)
> Bunun bir bedeli var. Dikdörtgende bu iki kutu aynı büyüklükte, ama gerçekte radardan uzaklaştıkça bir hücrenin kapladığı alan büyüyor: ekrandaki iki genişliğe bakın. Model bunu kendiliğinden bilemez; ileride CoordConv denemesinde bu noktaya geri döneceğiz.

### Slayt 3b · Üç ölçüm (statik)

**Ekranda:** DBZ · VEL · WIDTH panelleri (modelin gördüğü dikdörtgen) · renk çubukları · en güçlü dönmenin yeri halkayla
> Her pikselde üç ölçüm var. Yansıtıcılık yağışın ne kadar yoğun olduğunu, radyal hız rüzgârın radara yaklaşıp uzaklaştığını, spektrum genişliği o hücredeki hızların ne kadar dağınık olduğunu gösterir. Halka üç panelde de aynı yer: dönmenin en güçlü olduğu nokta. Yansıtıcılıkta kanca şekli, hızda yan yana iki renk, genişlikte bir parlama görüyoruz.

### Slayt 3c · Hortumun imzası (fırtınanın kareleri kadar kare; Boşluk ile oynar)

**Ekranda:** VEL'de dönüş çiftine yakın görünüm · dönen oklar · saat · TOR / EF · kare şeridi · 5 km ölçeği
> Hortumun radardaki imzası burada: yan yana iki bölge, biri radara yaklaşıyor, biri uzaklaşıyor. Bu dönüş çifti, modelden bulmasını beklediğimiz desen.
>
> (Boşluk) Aynı fırtınanın beş dakika arayla kaydedilmiş örnekleri. Görüntü çifti takip ediyor: fırtına ilerledikçe çift kaybolmuyor, onunla birlikte ilerliyor.
>
> Not: TorNet'te hortumlu olayların kareleri baştan sona hortum etiketli; hortumun oluşmadan önceki hali veri setinde ayrı bir örnek olarak yok.

---

## 4 · Modele ne veriyoruz? (2 statik slayt)

Sayılar 01 defterinin D bölümünün kayıtlı çıktılarından okunur (ham verinin 1.500 örneği üzerinde). Defter yeniden çalıştırılıp kaydedilince slaytlar da değişir. Seçim (kanallar, tarama, kare) `configs/data.json`'dan gelir.

### Slayt 4a · Hangi ölçüm?

**Ekranda:** 6 ölçüm için yatay AUC çubukları (siyah: hortum / hortumsuz, mor: hortum / yanlış alarm) · eksen 0,5'ten başlar · seçilmeyenler solar · "6 → 3" · seçilen 3 ölçüm ile 6 ölçümün birlikte AUC'si
> Ham veride her örnekte 6 farklı ölçüm var. Hepsini vermek yerine önce hangisinin işe yaradığına baktık: her ölçümden görüntünün merkezinde tek bir sayı çıkarıp hortumu ne kadar ayırdığını AUC ile ölçtük. 0,5 yazı tura demek.
>
> Radyal hız açık ara en iyisi, hem hortumu hortumsuzdan hem de en zor durumda, hortumu yanlış alarmdan ayırmada. Spektrum genişliği ikinci. Yansıtıcılık tek sayı olarak zayıf görünüyor ama asıl bilgisi şekilde, gördüğümüz kanca gibi; onu tek bir sayı ölçemez, CNN görebilir. O yüzden bu üçünü seçtik.
>
> Sağdaki iki sayı bunu doğruluyor: seçtiğimiz üç ölçüm birlikte, altısının hepsine çok yakın sonuç veriyor.

### Slayt 4b · Hangi açı, hangi an?

**Ekranda:** Seçilen kanallarda iki tarama açısının AUC'si · hortumun başladığı kare (1–4) · 6 ölçüm × 2 açı × 4 kare ızgarası sırayla söner, 3 kutu kalır · 1.382.400 → 86.400 · "16× küçük"
> Radar iki açıyla tarıyor. İkisi arasındaki fark küçük ama en önemli kanal olan hızda alçak tarama biraz daha iyi; fiziksel olarak da yere, yani hortuma en yakın olan o. Alçak taramayı seçtik.
>
> Her örnekte dört zaman adımı var ve etiket son kareye ait. Hortumların çoğu ancak son karelerde başlıyor; "şu an hortum var mı?" diye sorduğumuz için son kareyi aldık.
>
> Sağda bu üç kararın sonucu: altı ölçüm, iki açı, dört kare yerine üç kanal, bir açı, bir kare. Model bu üç kanallı, 120'ye 240'lık görüntüyü görüyor; veri on altı kat küçüldü.

---

## 5 · Ön işleme ve bölme (2 statik slayt)

### Slayt 5a · Ön işleme

**Ekranda:** Bölüm 3'teki fırtınanın üç kanalı: solda ham (boş pikseller damalı), sağda ağın girdisi (gri, −1…1) · ortada aralık ve boş piksel oranı → doldurma değeri · altta float32 → uint8 boyutu, en büyük yuvarlama hatası

Doldurma değerleri `tornet.from_uint8`'den, aralıklar `CHANNEL_RANGES`'ten, yuvarlama hatası 01 defterinin D6 çıktısından, boyut işlenmiş örnek sayısından gelir.

> Radar her yeri ölçemez: yağış yoksa yansıma da yoktur. Soldaki damalı yerler bu boş pikseller; bu örnekte ekrandaki oranlar kadar. Model "boş" bir değeri anlayamaz, o yüzden her kanalı anlamlı bir değerle dolduruyoruz: yansıtıcılık ve genişlikte en düşük değer, yani "yankı yok", "türbülans yok"; hızda sıfır, yani "hareket yok".
>
> Sonra her kanalı sabit bir fiziksel aralıkla −1 ile 1 arasına ölçekliyoruz; kanallar aynı büyüklükte olunca ağ birini ötekine tercih etmiyor. Aralıklar verimizden hesaplanmadığı için test kümesinden bilgi sızmıyor. Sağdaki gri görüntüler ağın gerçekte gördüğü sayılar.
>
> Saklarken her pikseli bir bayta indirdik: veri dört kat küçüldü. Kaybettiğimiz hassasiyet alttaki kadar; hortumdaki 20–60 m/s'lik hız farklarının yanında önemsiz.

### Slayt 5b · Bölme

**Ekranda:** Kümelerin oranı (çubuk) · her küme için örnek, fırtına, hortum oranı · "0 ortak fırtına", "10/10 kontrol" · sağda örnek fırtınalar: her satır bir fırtına, her nokta bir kare, renk kümesi

Sayılar `splits.csv`'den, kontroller 02 defterindeki `check_splits` ile canlı hesaplanır.

> Veriyi üçe böldük. Test kümesi veri setinin kendi test ayrımı; en sona kadar hiç dokunmadık. Doğrulama kümesi eğitimi izlemek, en iyi turu ve karar eşiğini seçmek için.
>
> Bölmeyi örneklere göre değil fırtınalara göre yaptık. Sağda her satır bir fırtına, her nokta onun bir karesi: bir fırtınanın bütün kareleri hep aynı renkte, yani aynı kümede. Aynı fırtınanın bir karesi eğitimde, öbürü testte olsaydı model ezberlediğini test ederdi.
>
> Gri noktalar çıkarılan örnekler: doğrulamadaki bir örneğe zamanda ve konumda çok yakın olan eğitim örneklerini attık. Sonuçta kümeler arasında ortak fırtına yok ve bütün kontroller geçiyor.

---

## 6 · TornadoCNN, hiperparametreler ve varyantlar (5 slayt)

Anlatılan model doğrulamada en iyi TornadoCNN varyantı (bugün CoordConv; adı slaytın sol üstünde). Varyantlar 6d'de başlangıç modeline (`03_baseline`) göre karşılaştırılır. Aktivasyonlar ve eğitilmemiş skor sunum açılırken torch ile hesaplanır (eğitilmemiş ağ: aynı mimari, rastgele ağırlık, seed 0); eğitilmiş ağ `best.pt`. Örnek, bölüm 3'teki fırtınanın en güçlü karesi.

### Slayt 6a · İleri geçiş, adım adım (19 kare, eğitilmemiş ağ)

**Ekranda:** Üstte örneğin ağdaki yolculuğu (her blokta en güçlü 8 harita). Altta işlem şeması: girdi · 4 blok (Conv 3×3 → ReLU → MaxPool 2×2) · sınıflandırıcı (Global Max → Dense 64 → ReLU → Dropout → Dense 1 → Sigmoid). Etkin işlem turuncu; yanında küçük bir çizim, çıktının boyutu ve tek satırlık açıklama. Her kare tek bir işlem.

Renkler: Conv çıktısı işaretli (mavi negatif, kırmızı pozitif); ReLU ve sonrası 0'dan en büyüğe.

**Girdi**
> İlk modelimiz TornadoCNN: TorNet makalesindeki modelden uyarlayıp sıfırdan yazdığımız küçük bir evrişimli ağ. Birkaç farklı sürümünü denedik; burada doğrulamada en iyi sonucu veren sürümü anlatıyoruz, ötekileri birazdan göstereceğim. Girdisi az önce hazırladığımız üç kanallı görüntü. (CoordConv ise:) Bu sürüm üç radar kanalına iki kanal daha ekliyor: her pikselin radara uzaklığı, r ve 1/r. Bölüm 3'teki piksel büyüklüğü sorununu hatırlayın; model artık hangi pikselin ne kadar uzakta olduğunu biliyor. Şimdi bu girdinin ağın içinden nasıl geçtiğini işlem işlem izleyeceğiz.

**Conv 3×3** (her blokta ilk kare)
> Evrişim: 3'e 3'lük küçük bir filtre görüntünün üzerinde kayıyor ve her noktada komşuluğun ağırlıklı toplamını alıyor. Bu blokta ekrandaki sayıda filtre var; her biri ayrı bir özellik haritası üretiyor. Boyut değişmiyor. Değerler negatif de olabiliyor: mavi negatif, kırmızı pozitif.

**ReLU**
> Aktivasyon: negatif değerler sıfıra iniyor, pozitifler aynen kalıyor. Mavilerin kaybolduğuna dikkat edin. Ağın doğrusal olmayan ilişkileri öğrenebilmesi bu adım sayesinde.

**MaxPool 2×2**
> Havuzlama: her 2'ye 2'lik pencereden yalnızca en büyük değer kalıyor; harita yarıya küçülüyor. Ayrıntı azalıyor ama en güçlü sinyal korunuyor ve sonraki filtreler daha geniş bir alanı görüyor.

(Bloklar 2–4 aynı üç adım: filtre sayısı ikiye katlanıyor, boyut yarıya iniyor. Burada hızlı geçilebilir.)

**Global Max**
> Son bloktaki her haritadan yalnızca en güçlü değeri alıyoruz: 128 sayı. Hortum görüntünün neresinde olursa olsun sinyali kaybolmadan aktarılıyor.

**Dense 64 → ReLU**
> Bu 128 sayı tam bağlantılı bir katmanda 64 sayıya dönüşüyor, ardından yine ReLU.

**Dropout**
> Eğitimde bu nöronların bir kısmı her adımda rastgele kapatılıyor; model tek bir nörona yaslanamıyor, ezber azalıyor. Tahmin yaparken hepsi açık.

**Dense 1 → Sigmoid**
> Son katman tek bir sayı veriyor; sigmoid onu 0 ile 1 arasına sıkıştırıyor: skor. Ama ağ henüz hiçbir şey öğrenmedi; ağırlıklar rastgele, skor da anlamsız. Şimdi bu ağırlıkları nasıl öğrettiğimize bakalım.

### Slayt 6b · Eğitim (epoch sayısı kadar kare; Boşluk ile oynar)

**Ekranda:** Tur sayacı · kayıp (kesikli: eğitim, düz: doğrulama) · doğrulama PR-AUC · en iyi tur işareti · erken durdurma bölgesi · kayıp ağırlığı (× pos_weight)
> Dengesizliğe karşı kayıp fonksiyonunda her hortum örneğinin hatasını soldaki kat kadar ağır saydık; böylece model nadir sınıfı görmezden gelemiyor.
>
> (Oynat) Eğitim kaybı her turda düşüyor. Doğrulama başarısı bir noktaya kadar yükseliyor, sonra düşmeye başlıyor: model eğitim verisini ezberlemeye başladı. İşaretli turda en iyi değere ulaştı; sonraki turlarda iyileşme olmayınca eğitimi durdurduk ve en iyi turun ağırlıklarını sakladık.

### Slayt 6c · Eğitimden önce / sonra (statik)

**Ekranda:** VEL (dönüş çifti halkalı) · 3. bloğun en güçlü 4 haritası, önce ve sonra · skorlar ve eşik
> Aynı fırtına, aynı katman. Eğitimden önce haritalar her yerde rastgele parlıyor. Eğitimden sonra yalnızca halkanın içi, yani dönüş çiftinin olduğu yer parlıyor: model hortumun imzasına bakmayı kendisi öğrendi. Skor da ekrandaki değere çıktı; eşiğin üstünde olduğu için model "hortum" diyor. Eşiği doğrulama kümesinde seçtik.

### Slayt 6d · Hiperparametre denemeleri (statik, tek grafik)

**Ekranda:** 15 deneme, gruplara göre (optimizer, model boyutu, kayıp ağırlığı, batch, düzenlileştirme, öğrenme oranı); her satırda tek bir ayar değişti. Doğrulama PR-AUC'nin baseline'a (3 seed ortalaması) göre farkı; gri bant seed aralığı. Yeşil: bandın üstü, kırmızı: altı, gri: fark yok. SGD eksenin dışında kaldığı için okla. Sağda iyileşme / fark yok / kötüleşme sayıları.

Değişiklikler her denemenin `results.json` ayarlarının baseline'la karşılaştırılmasından otomatik çıkarılır (en fazla tur sayısı bütün denemelerde 30, sonucu etkilemez). Defterdeki kural (H2) ±0,02 gürültü sınırıydı; slayt ise seed bandını kullanıyor, sınırdaki denemelerde ikisi farklı sonuç verebilir.

> Modeli kurduktan sonra önce ayarlarını denedik: her seferinde yalnızca birini değiştirdik. Çoğu denemenin noktası gri bandın içinde, yani fark aynı modeli yeniden eğittiğimizde gördüğümüz oynamadan büyük değil.
>
> Asıl öğrendiğimiz kırmızılar: düz SGD hiç öğrenemedi, öğrenme oranını düşürmek ve dropout'u kaldırmak belirgin şekilde kötüleştirdi; yani varsayılan Adam, 0,001 öğrenme oranı ve dropout yerinde.
>
> Ekrana göre söyle: bandın biraz üstüne çıkan denemeler varsa (bugün batch 128 ve L2 = 0,001) farkları çok küçük ve tek seed; defterdeki ±0,02 kuralını geçmiyorlar, bu yüzden varsayılan ayarları koruduk. Sonra ayarlar yerine mimaride değişiklikler denedik.

### Slayt 6e · Varyantlar (her varyant bir kare)

**Ekranda:** Varyantın adı ve değişikliği gösteren küçük animasyon · sağda doğrulama PR-AUC'nin baseline'a (3 seed ortalaması) göre farkı · gri bant: baseline'ın seed aralığı

Test sonuçları burada yok; 8. bölümde.

> Sonra modelde her seferinde tek bir şeyi değiştirip etkisini ölçtük. Gri bant önemli: aynı modeli yalnızca rastgele başlangıcı değiştirerek üç kez eğittiğimizde sonuç bu kadar oynuyor. Bandın içinde kalan bir fark gerçek bir iyileşme sayılmaz.
> - **Global Avg:** En güçlü değer yerine ortalama alınca, küçük bir hortumun tek bir yerdeki güçlü sinyali sulanıyor.
> - **BatchNorm:** Her katmanın çıktısını ortalayıp ölçekliyoruz.
> - **CoordConv:** Modele her pikselin radara uzaklığını iki ek kanal olarak veriyoruz; bölüm 3'te gördüğümüz piksel büyüklüğü sorununa çözüm.
> - **Augmentation:** Eğitimde görüntüyü biraz kaydırıp gürültü ekliyoruz; ayna ya da döndürme yok, çünkü hız ve kutupsal ızgara bunlara izin vermiyor.
> - **Label smoothing:** Hortum hedefini kesin 1 yerine biraz aşağı çekiyoruz.
> - **TorNet ağırlıkları:** Sınıf ağırlığı yerine TorNet'in kategori ağırlıkları; uyarılar yarım, güçlü hortumlar iki kat.
>
> Yıldızlı olan seçtiğimiz varyant: doğrulamada en iyisi, bu yüzden sunumun başından beri anlattığımız model o. Dürüst olmak gerekirse çoğu varyant bandın içinde; seçtiğimiz de tek seed'le eğitildi ve farkı seed aralığına yakın. Bu yüzden yeniden eğitimde onu da üç seed'le eğiteceğiz.

---

## 7 · ResNet-18: baseline ve ön-eğitim (5 slayt)

ResNet ailesinin baseline'ı **sıfırdan (rastgele ağırlıklarla) eğitilen ResNet-18**; ön-eğitim (ImageNet ağırlıklarıyla başlamak) onun üzerine denediğimiz varyant. Akış bölüm 6 ile aynı: önce mimari (rastgele ağırlıklarla, yani baseline'ın başlangıcı), sonra iki başlangıcın karşılaştırması ve seçim, sonra seçilenin eğitimi ve önce/sonra, en sonda üç modelin karşılaştırması. Bundan sonraki bütün slaytlarda sıfırdan model "(baseline)" diye anılır. Sayılar `results.json`, `history.json` ve `preds.parquet`'ten; yalnız doğrulama kümesi.

### Slayt 7a · ResNet-18 içinden geçiş (10 kare, rastgele ağırlıklar)

**Ekranda:** Üstte aşamaların çıktıları, altta işlem şeması: girdi · kök (Conv 7×7 /2 → BN → ReLU → MaxPool 3×3 /2) · 4 aşama (her biri 2 residual blok) · Global Avg → Dense 1 → Sigmoid. Residual aşamalarında küçük çizim: F(x) + x.

**Girdi**
> İkinci model ailemiz ResNet-18: görüntü tanıma için tasarlanmış hazır ve çok daha derin bir ağ. Önce mimarisini, TornadoCNN'de yaptığımız gibi rastgele ağırlıklarla, yani hiçbir şey öğrenmemiş haliyle gösteriyorum; bu aynı zamanda ResNet baseline'ımızın başlangıç noktası.

**Conv 7×7 /2 · BN → ReLU · MaxPool 3×3 /2** (kök)
> Daha büyük, 7'ye 7'lik filtrelerle başlıyor ve iki adımda bir kaydırdığı için boyut daha ilk katmanda yarıya iniyor. Arkasından normalleştirme ve ReLU, sonra bir havuzlama daha: görüntü daha ilk aşamada dörtte birine küçüldü.

**Aşama 1–4** (residual)
> Asıl fark burada. Her aşamada iki residual blok var: girdi iki evrişimden geçerken bir kopyası da yanından dolaşıyor ve sonuca ekleniyor, F(x) + x. Böylece her blok yalnızca "farkı" öğreniyor ve çok derin ağlar eğitilebiliyor. Her aşamada filtre sayısı ikiye katlanıp boyut yarıya iniyor: sonunda 512 harita, her biri yalnızca 4'e 8.

**Global Avg · Dense 1 → Sigmoid**
> TornadoCNN en güçlü değeri alıyordu; ResNet ortalamayı alıyor. Ağırlıklar rastgele olduğu için skor henüz anlamsız.

### Slayt 7b · Aynı ağ, iki başlangıç: baseline ve ön-eğitim (statik)

**Ekranda:** Solda ResNet baseline (rastgele ağırlıklarla başla, sıfırdan), sağda varyant (ImageNet ağırlıklarıyla başla); her kartta ilk katmanın filtreleri başlangıçta ve radarla eğitimden sonra, altında doğrulama PR-AUC. Ortada ön-eğitimin katkısı ve seed'ler arası oynama. Daha iyi olan kart turuncu çerçeveli.
> Bu ağı iki şekilde eğittik. Soldaki ResNet baseline'ımız: rastgele ağırlıklarla, sıfırdan eğittik. Başlangıç filtreleri gürültü ve elimizdeki veriyle eğitim onları anlamlı filtrelere çeviremiyor. Sağdaki varyantta aynı ağı ImageNet ağırlıklarıyla başlattık: filtreler en başından düzenli kenar ve renk dedektörleri ve eğitimden sonra da öyle kalıyorlar.
>
> Ortadaki sayı ön-eğitimin katkısı; doğrulamada seed'ler arası oynamadan büyük. Yani ResNet ailesinde ön-eğitim işe yarıyor ve bu yüzden ResNet olarak ön-eğitimliyi seçtik; şimdi seçtiğimiz bu modelin eğitimine ve içine bakalım.

### Slayt 7c · Eğitim (tur sayısı kadar kare; Boşluk ile oynar)

**Ekranda:** Tur sayacı · ResNet'in kaybı (kesikli: eğitim, düz: doğrulama) · doğrulama PR-AUC (turuncu seçilen ResNet, kesikli mavi ResNet baseline, gri TornadoCNN) · en iyi tur · erken durdurma
> Öğrenme oranını küçük tuttuk ki ImageNet'ten gelen bilgi bozulmasın. (Oynat) Eğitim kaybı çok hızlı düşüyor, ama doğrulama kaybı daha ikinci turdan sonra fırlıyor: ResNet eğitim verisini ezberliyor. En iyi turu çok erken, ikinci turda; kesikli mavi baseline daha düşükten başlıyor; gri çizgiyle karşılaştırın, küçük CNN aynı turlarda ikisini de geçiyor. Bir turu da üç kat uzun sürüyor.

### Slayt 7d · Yalnız ImageNet / ImageNet + radar (statik)

**Ekranda:** Sol üstte model adı · iki satır aynı model (ResNet-18 · ön-eğitimli): üstte radar görmeden önce ("yalnız ImageNet"), altta radarla eğitildikten sonra ("ImageNet + radar") · VEL (dönüş çifti halkalı) · aşama 3'ün en güçlü 4 haritası · skor ve eşik · ilk katmanın 64 filtresi
> İki satırda da aynı model var; fark yalnızca radar verisiyle eğitilip eğitilmediği. Radar görmeden önce aşama 3'ün haritaları görüntünün kenarlarına ve rastgele yerlere tepki veriyor. Radarla eğittikten sonra dönüş çiftinin olduğu yer parlıyor ve skor eşiğin üstüne çıkıyor.
>
> Sağdaki küçük kareler ilk katmanın filtreleri: fotoğraflardan öğrenilmiş kenar ve renk dedektörleri. Eğitimden sonra neredeyse aynılar; yani ImageNet'in bu temel filtrelerini radar için olduğu gibi yeniden kullandık.

### Slayt 7e · TornadoCNN mi ResNet mi? (statik)

**Ekranda:** Parametre sayısı (kare alanı) · tur süresi · doğrulama PR-AUC, her nokta bir seed · son katmanın en güçlü haritası aynı fırtınada: 7 × 15 ve 4 × 8 ızgara
> Bunlar sunumun geri kalanında ve arayüzde kullandığımız üç model: seçtiğimiz CNN, seçtiğimiz ResNet ve karşılaştırma için ResNet baseline'ı. ResNet'in yüz kat fazla parametresi var ve her turu üç kat uzun sürüyor. Buna rağmen doğrulamada küçük CNN'in gerisinde.
>
> Sağdaki iki harita nedenini de gösteriyor: ResNet kararını 4'e 8'lik çok kaba bir ızgarada veriyor; bir hücre kilometrelerce alanı kaplıyor. Küçük bir dönüş çifti için TornadoCNN'in daha ince ızgarası daha uygun. Bizim verimiz için daha büyük model daha iyi model değil.

---

## 8 · Test sonuçları (4 slayt)

Yalnız temel 3 model; baseline (3 seed) yalnızca 8b'de referans bant. Test kümesi bu bölüme kadar hiç kullanılmadı. Güven aralıkları fırtına bazlı bootstrap (500 örneklem); "fark / eğilim / fark yok" etiketini Karşılaştırma sekmesindeki aynı kural üretir. İlk açılışta hesap ~10 sn sürer, sonra önbellekten gelir.

### Slayt 8a · Eğriler (2 kare, çizilerek gelir)

**Ekranda:** Test kümesinin boyutu (kare · fırtına · hortum) · ROC eğrileri ve ROC-AUC'ler · sonra PR eğrileri, rastgele tahmin çizgisi ve PR-AUC'ler
> Şimdi en baştan ayırdığımız ve bugüne kadar hiç dokunmadığımız test kümesi: ekrandaki kadar kare, fırtına ve hortum. ROC eğrisi, eşiği değiştirdikçe yakalanan hortumla yanlış alarm arasındaki ödünleşimi gösterir; üç model de köşegenin, yani rastgele tahminin çok üstünde.
>
> (→) Dengesiz veride daha dürüst olan PR eğrisi: alttaki kesikli çizgi rastgele tahmin, yani hortum oranı. Sıralama doğrulamadakiyle aynı: önce CNN, sonra ön-eğitimli ResNet, en sonda sıfırdan ResNet.

### Slayt 8b · Ne kadar eminiz? (statik)

**Ekranda:** Üç modelin test PR-AUC'si ve %95 güven aralığı · her satırda kendi ailesinin baseline'ı: CNN için gri bant (3 seed) ve kesikli ortalama, ön-eğitimli ResNet için kesikli çizgi (ResNet baseline'ı, sıfırdan) · sağda kendi baseline'ına göre fark ve etiket; sıfırdan ResNet'in kendisi "ResNet baseline"
> Test kümesi sınırlı olduğu için her sonucun bir belirsizliği var; çubuklar bunu gösteriyor. Her modeli kendi ailesinin baseline'ıyla karşılaştırıyoruz: seçtiğimiz CNN varyantını CNN baseline'ıyla, ön-eğitimli ResNet'i sıfırdan eğitilen ResNet'le.
>
> Ekrana göre söyle: ikisi de testte kendi baseline'ının üstünde, yani seçimlerimiz testte de aynı yönde. Ama güven aralıkları sıfırın hemen altından başladığı için bunlar "eğilim" ya da "kesin değil"; kesin bir üstünlük iddia edemiyoruz. Aileler arasında ise fark net: CNN'in aralığı ResNet'lerinkinden belirgin şekilde yukarıda.

### Slayt 8c · Karar eşiğinde (statik)

**Ekranda:** Üç modelin karışıklık matrisi yan yana (eşikler doğrulamada seçildi; sayılar sayarak gelir) · her matrisin altında yakalama (POD), isabet (SR) ve CSI
> Her model kendi eşiğinde: seçtiğimiz CNN ekrandaki kadar hortumu yakaladı, bu kadarını kaçırdı ve bu kadar yanlış alarm verdi; yani hortumların yaklaşık yarısını yakalıyor ve "hortum" dediği her iki durumdan biri gerçekten hortum.
>
> ResNet'ler benzer sayıda hortum yakalıyor ama çok daha fazla yanlış alarm veriyor; özellikle sıfırdan eğitilen ResNet'te yanlış alarm neredeyse iki katı. CSI, yani yakalama ile yanlış alarmı birlikte ölçen tek sayı, bunu özetliyor.

### Slayt 8d · Zorluğa göre (statik)

**Ekranda:** PR-AUC üç görünümde (hepsi · hortum / sıradan fırtına · hortum / uyarı), altında hortum oranı · hortum şiddetine göre (EF0, EF1, EF2, EF3+) eşikte yakalama oranı, altında kare sayısı
> Soruyu zorlaştıralım. Hortumu sıradan fırtınalardan ayırmak kolay; asıl zor olan, uyarı verilmiş ama hortum çıkmamış fırtınalardan ayırmak. CNN bu en zor durumda da önde.
>
> Sağda hortumun şiddeti: güçlü hortumların büyük kısmını yakalıyoruz, zayıf EF0 hortumların ise yarısından azını. Zayıf hortumların dönüşü bu çözünürlükte çok küçük; modelin asıl zorlandığı yer burası.

---

## 9 · Model neye bakıyor? · Kapanış (7 slayt)

Temel 3 model. Her yöntemin önce "nasıl çalışır" slaytı gelir (seçilen CNN, bölüm 3'ün karesi, gerçek ara değerler; `src/helper/xai.py` ile, Açıklanabilirlik sekmesiyle aynı yöntem), sonra sonuç slaytı. Fırtınadaki Grad-CAM ve "nasıl" slaytlarının değerleri sunum açılırken torch ile hesaplanır; kanal payları ve toplu açıklama testleri önceden hesaplanmış dosyalardan okunur (`scripts/make_xai.py`, `scripts/make_xai_global.py`).

### Slayt 9a · Grad-CAM nasıl hesaplanır? (5 kare, altta adım şeması)

**Ekranda:** 1 son evrişim katmanının çıktısı (128 harita, 15 × 30; ağırlığı en büyük 8'i) → 2 her haritanın ağırlığı (128 çubuk, sıralı; turuncu hortum lehine, mavi aleyhine) → 3 ağırlıklı toplam (işaretli) → 4 ReLU (yalnız pozitif) → 5 120 × 240'a büyütülüp VEL'in üzerine bindirilmiş Grad-CAM
> Modelin kararını neye dayandırdığını görmek için Grad-CAM kullandık. Nasıl çalıştığını aynı fırtına üzerinde adım adım gösteriyorum.
>
> (1) Son evrişim katmanında 128 özellik haritası var; her biri bir desene tepki veriyor. (2) Her harita için şunu soruyoruz: bu harita biraz artsa skor ne kadar artar? Bu, skorun haritaya göre türevi, yani gradyanı; ortalamasını o haritanın ağırlığı yapıyoruz. Turuncular hortum lehine, maviler aleyhine. (3) Haritaları ağırlıklarıyla topluyoruz. (4) Yalnızca hortum lehine olan kısmı bırakıyoruz. (5) Bu kaba haritayı görüntü boyutuna büyütüp radar görüntüsünün üzerine koyuyoruz: model en çok halkanın, yani dönüş çiftinin olduğu yere bakıyor.

### Slayt 9b · Üç modelin Grad-CAM'i (fırtınanın kareleri kadar kare; Boşluk ile oynar)

**Ekranda:** Bölüm 3'teki fırtına, üç modelin Grad-CAM'i yan yana VEL üzerinde · halka: dönüş çifti, kırmızı artı: Grad-CAM tepesi · altlarında tepenin çifte uzaklığı (km) · saat ve kare şeridi
> Aynı hesabı üç modelde ve fırtınanın bütün karelerinde yaptık. Halka dönüş çifti, artı modelin en çok baktığı yer.
>
> (Oynat) CNN'in dikkati dar bir bölgede ve fırtına ilerlerken dönüş çiftini takip ediyor. ResNet'lerin haritası ise neredeyse bütün görüntüyü kaplıyor: 4'e 8'lik kaba ızgarada karar veriyorlar, kararın tam olarak nereden geldiği belli değil.

### Slayt 9c · Bir kanalın katkısı nasıl ölçülür? (9 kare)

**Ekranda:** Solda üç kanal (kapalı olan solar), ortada o birleşimdeki gerçek skor, sağda 8 birleşimin listesi kare kare dolar · son karede şelale: hiç kanal yokken skor → +DBZ → +VEL → +WIDTH = gerçek skor
> Hangi ölçümün karara ne kadar katkı verdiğini bulmak için kanalları açıp kapattık. Üç kanalın sekiz birleşimi var; her birinde modelin skoruna bakıyoruz. Kapalı kanal "ölçüm yok" demek.
>
> (Kareler) Ekrana göre söyle: bu karede yansıtıcılık tek başına da, hız tek başına da hortum dedirtmiyor; ikisi birlikte olunca skor neredeyse 1'e çıkıyor. Model iki kanalı birlikte kullanıyor.
>
> (Son kare) Shapley değeri, her kanalın bütün birleşimler üzerinden ortalama katkısı; katkılar toplanınca tam olarak gerçek skoru veriyor. Bu karede en büyük katkı hızın, sonra yansıtıcılığın.

### Slayt 9d · Hangi kanal? (bütün test kümesi) (statik)

**Ekranda:** Her model için kararın kanallara dağılımı (kanal SHAP, bütün test kümesi): "hortum dedi · doğru" ve "yok dedi · doğru" çubukları · sağda VEL payının değişimi
> Hangi ölçümün karara ne kadar katkı verdiğini de ölçtük. Model "hortum yok" derken radyal hızın payı düşük; "hortum" derken belirgin şekilde artıyor. Yani modeller hortum kararını gerçekten dönüşü taşıyan kanala dayandırıyor; bölüm 4'teki seçimimizle de tutarlı.

### Slayt 9e · Silme testi nasıl yapılır? (6 kare: %0 → %50)

**Ekranda:** Aynı kare iki kez: solda Grad-CAM'in önemli dediği pikseller önce, sağda rastgele pikseller siliniyor (silinenler damalı) · altlarında skor · sağda silinen orana göre iki skor eğrisi kare kare uzar
> Bir açıklamanın doğru olup olmadığını nasıl anlarız? Modelin önemli dediği pikselleri silip skora bakarız. Açıklama doğruysa skor hızla düşmeli.
>
> (Kareler) Ekrana göre söyle: Grad-CAM'in gösterdiği bölgeden piksellerin küçük bir kısmını silmek skoru hemen düşürüyor; aynı sayıda rastgele piksel silmek ise skoru neredeyse hiç değiştirmiyor. Yani Grad-CAM gerçekten modelin kullandığı yeri gösteriyor.

### Slayt 9f · Açıklamaya güvenebilir miyiz? (üç model, toplu) (statik)

**Ekranda:** Sol: silme testi, Grad-CAM'in önemli dediği pikseller silinince skor rastgele sıraya göre kaç kat hızlı / yavaş düşüyor (ortadaki çizgi rastgele) · sağ: doğru yakalanan hortumlarda Grad-CAM tepesinin dönüş çiftine ortanca uzaklığı
> Bu haritalara körü körüne güvenmedik, test ettik. Modelin önemli dediği pikselleri sildiğimizde skoru gerçekten hızla düşüyorsa, açıklama modelin kullandığı yeri gösteriyor demektir.
>
> Ekrana göre söyle: CNN'in açıklaması rastgeleden belirgin şekilde hızlı, yani güvenilir; ve tepesi dönüş çiftine birkaç kilometre yakın. ResNet'lerde Grad-CAM rastgeleden bile yavaş: kaba ızgara yüzünden bu modellerde Grad-CAM'e güvenemiyoruz. Bu da 7. bölümdeki "ResNet kaba bakıyor" bulgusunu bağımsız bir yoldan doğruluyor.

### Slayt 9g · Kapanış (statik)

**Ekranda:** 4 sayı: örnek sayısı ve yıllar · seçilen CNN'in test PR-AUC'si · eşikte yakalanan hortum oranı · güçlü (EF3+) hortumlarda yakalama · sınırlamalar · uygulamanın sekmelerine giden kartlar (tıklayınca o sekme açılır)
> Özetle: radar verisini kutupsal haliyle, sızıntısız bölerek modele verdik; küçük, kendi yazdığımız bir CNN'i büyük, hazır bir ResNet'le karşılaştırdık ve küçük model her ölçütte önde çıktı. Test kümesinde hortumların yarısından fazlasını, güçlü hortumların neredeyse hepsini yakalıyor ve kararını dönüşe bakarak veriyor.
>
> Sınırlamalarımız da belli: yalnızca üç yıl veri, tek tarama açısı ve tek kare kullandık; zayıf hortumlarda zorlanıyoruz ve seçtiğimiz varyant tek seed'le eğitildi.
>
> (Kartlardan birine tıkla) Uygulamanın geri kalanında her fırtınayı tek tek inceleyebilir, modellerin hatalarına ve haritaya bakabilirsiniz.
