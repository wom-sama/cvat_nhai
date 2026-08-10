# CVAT Nhai

Ung dung desktop gan lai nhan anh nhanh. Tool ho tro bo du lieu xoai 5 lop
YOLO/classification va che do phan loai thu muc don gian voi class dong. Trong
che do gan nhan bbox, moi lan nhan `Enter`, tool tao dong thoi:

- Anh day du va label YOLO trong `D:\DataAI\AIEx\dataset`.
- Crop classification trong
  `D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops`.
- Dong moi trong `manifest.csv`.
- Cap nhat `stats.json` va `canbang.yaml`.
- Di chuyen anh nguon vao safety archive sau khi tat ca output da ghi thanh cong.

## Nam lop

| ID | Ten thu muc / ten trong YAML |
|---:|---|
| 0 | `Xoai_Song_Chua_KhoDap` |
| 1 | `Xoai_Song_ChuaNhe_CoNguyCo` |
| 2 | `Xoai_Chin_NgotThanh_DeDap` |
| 3 | `Xoai_ChinGia_NgotGat_KhongVanChuyen` |
| 4 | `Xoai_Hu_KhongAnDuoc` |

Neu dataset detection dang dung schema 4 lop gop class 0 va 1, ung dung
se khoa thao tac ghi cho den khi migration 5 lop hoan tat. Nut
`Chuyen dataset sang 5 lop`:

1. Doi chieu moi dong YOLO voi `cls_crops/manifest.csv`.
2. Dung neu co bat ky anh nao khong khop.
3. Tao ZIP backup trong `dataset/_cvat_nhai_backup`.
4. Chi sau do moi ghi ID `0..4` va cap nhat `data.yaml`, `canbang.yaml`.

Dataset tren may ban giao da duoc migration ngay 06/06/2026. Backup truoc
migration nam tai:

`D:\DataAI\AIEx\dataset\_cvat_nhai_backup\before_5class_20260606_172923.zip`

## Chay

Nhan dup `run.bat`, hoac:

```powershell
cd D:\DataAI\AIEx\CVAT_
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m cvat_nhai.app
```

## Thao tac

- `Thu muc dich`: chon mot thu muc goc. Tool tu tao
  `<dich>\dataset` cho YOLO, `<dich>\cls_crops` cho classification va
  `<dich>\.cvat_nhai_archive` cho hoan tac.
- `1` den `5`: chon class.
- Keo chuot trai: ve bbox.
- Keo trong bbox: di chuyen.
- Keo 8 diem vuong: thay doi kich thuoc.
- Con lan chuot: zoom.
- `Space` + keo hoac chuot giua: pan.
- Double click: fit anh.
- `Enter`: ghi hai dataset, cap nhat metadata, xoa anh khoi thu muc nguon.
- `Delete`: loai anh ma khong ghi vao dataset.
- `F`: xoa bbox va class hien tai de lam lai.
- `Ctrl+Z`: hoan tac thao tac ghi/xoa gan nhat.
- `A` / `D`: xem anh truoc / sau ma chua xu ly.

## Xem va sua dataset YOLO cu

Chon che do `Xem / sua dataset YOLO cu`, sau do chon thu muc co
`data.yaml`. Tool doc cac split trong YAML, ghep `images` voi `labels` va
hien thi tat ca bbox/class da co.

- Click mot box de chon object.
- Keo box hoac 8 handle de move/resize.
- Nhan `1` den `5` de doi class cua object dang chon.
- Keo tren vung trong de them object moi.
- `Backspace` xoa rieng object dang chon trong bo nho.
- `Enter` backup label cu, ghi label moi theo kieu atomic va sang anh tiep.
- `F` bo moi thay doi chua luu va tai lai snapshot nhan da co.
- `Delete` chuyen ca anh va label vao
  `<dataset>\.cvat_nhai_editor_archive`.
- `Ctrl+Z` bo thay doi chua `Enter`; neu anh da duoc ghi hoac xoa thi khoi
  phuc label/anh va `canbang.yaml` tu transaction gan nhat.
- `A` / `D` chi chuyen anh khi thay doi hien tai da duoc `Enter` hoac `F`,
  tranh mat nhan dang sua.
- Thanh tien do co the click hoac keo nhanh toi bat ky anh nao. Trong luc
  keo chi cap nhat so thu tu; anh chi duoc tai mot lan khi tha chuot.
  `Left/Right` di mot anh, `PageUp/PageDown` nhay 100 anh. Thanh se tu quay
  lai vi tri hien tai neu anh dang co thay doi chua `Enter` hoac `F`.

Editor chi nhan YOLO bounding-box 5 cot:

```text
class_id center_x center_y width height
```

Nhan segmentation/polygon se bi tu choi va khong bi ghi de.
BBox duong nhung nho hon 3 pixel van duoc nap; UI ve marker toi thieu de co
the click/xoa, trong khi toa do YOLO goc duoc giu nguyen. Khi mot anh co
nhieu object, box co dien tich lon nhat duoc chon mac dinh.

Nut `Xuat yolo_f + class_f` chon mot thu muc goc rong va tao:

```text
<export>\yolo_f\images\<split>\*
<export>\yolo_f\labels\<split>\*.txt
<export>\class_f\<split>\<class_name>\*_boxNNN.jpg
```

Ca hai bo dung cung mot split moi theo ty le 70/20/10. Thuat toan can bang
so object cua tung class thay vi chi chia tong so anh. Tat ca bbox cua cung
mot anh, anh co cung family name va anh trung/noi dung gan trung theo visual
fingerprint duoc khoa trong cung mot `leakage_group`, nen khong the nam dong
thoi o train va val/test. `manifest.csv` luu split nguon, split moi va group
de audit data leak.

Moi bo co `data.yaml`, `manifest.csv` va `canbang.yaml`; `class_f` co them
`stats.json`. Thu muc export bat buoc phai rong; ket qua duoc dung trong thu
muc tam va chi duoc dua vao dich sau khi toan bo anh, nhan va crop thanh
cong. Thu muc dich phai nam ngoai dataset nguon. Dataset YOLO nguon khong
bi di chuyen hay sap xep lai.

Khi export, ung dung hien dialog gom 4 giai doan, ten anh dang xu ly va thanh
tien do. Nut `Huy an toan` dung tai diem an toan va xoa thu muc tam; dataset
nguon khong bi thay doi va khong de lai bo export dang do.

Khi nhan `Enter`, `DEL` hoac hoan tac, tac vu chay ngoai UI thread. Thao tac
nhanh khong lam chop overlay; neu vuot 180 ms, overlay `Dang xu ly` tu hien de
nguoi dung biet app van dang lam viec. Khi anh ke tiep dang preload, canvas giu
anh hien tai kem thong bao tai thay vi chuyen sang man hinh trong.

Truoc khi chon thu muc xuat, dialog `Cau hinh class_f` cho phep chinh margin
tu `0%` den `50%` moi canh bbox. Bon crop ngau nhien duoc cap nhat truc tiep
khi keo slider hoac sua o so; nut `Doi mau ngau nhien` chon cac bbox khac.
Gia tri margin export duoc ghi nho rieng cho lan sau.
Nut `Tiep tuc chon thu muc xuat` mo bo chon thu muc Qt ngay tren cua so chinh,
tranh truong hop hop thoai native bi an phia sau tren Windows.

Trong che do sua, `Tab` chon box ke tiep va `Shift+Tab` chon box truoc do,
ke ca khi focus dang nam tren sidebar.

Che do sua hien `Split hien tai` cua anh dang xem va co bo loc `Tat ca`,
`train`, `val`, `test` ngay canh thanh tien do. Ben duoi la bo loc class theo
nhan YOLO da luu tren dia; anh co nhieu bbox se xuat hien trong tat ca class co
trong label. Hai bo loc nay chi doi danh sach anh dang duyet de dieu tra data
leak; export van dung toan bo dataset YOLO da chinh sua tren dia.

## Kiem tra va sua class_f

Chon che do `Kiem tra / sua class_f`, sau do chon thu muc `class_f` hoac thu
muc cha co thu muc con `class_f`. Tool doc `data.yaml` neu co, neu khong se
dung schema 5 lop mac dinh, roi quet layout:

```text
class_f\<split>\<class_name>\*.jpg
```

Moi anh crop duoc hien kem split va class hien tai suy ra tu thu muc. Chon
class moi bang phim `1` den `5` hoac nut class, nhan `Enter` de ap dung. Tool
chi di chuyen anh trong cung split sang thu muc class moi, khong doi train/val/
test, va cap nhat `manifest.csv`, `canbang.yaml`, `stats.json`, `data.yaml`.
Neu ten file da ton tai trong class dich, tool tao ten phu `__classeditNNN`
thay vi ghi de.

`F` dua class dang chon ve class goc cua anh hien tai. `Delete` dua anh crop
vao `class_f\.cvat_nhai_classification_archive` va xoa dong tu manifest neu
co. Khi class chua luu, app chan chuyen anh, keo thanh tien do va doi bo loc de
tranh lam roi anh vao class sai do thao tac nhanh.

`Ctrl+Z` bo lua chon class chua luu. Sau `Enter` hoac `Delete`, phim nay phuc
hoi dung duong dan anh, class, dong `manifest.csv`, `canbang.yaml` va
`stats.json`. Undo tu choi ghi de neu file dich hoac manifest da bi sua ben
ngoai ung dung.

Sau khi mo dataset, tool giu chi muc manifest va bo dem split/class trong bo
nho. Moi thay doi chi cap nhat cac class lien quan, khong quet lai toan bo cay
`class_f`; manifest va metadata van duoc ghi atomic, co rollback neu mot buoc
that bai.

Che do nay dung chung bo loc split va class voi YOLO editor. Bo loc class_f dua
tren class thu muc hien tai cua anh crop; sau khi `Enter` hoac `Delete`, danh
sach dang xem duoc tinh lai ngay.

Crop classification duoc mo rong theo `crop_padding`, sau do resize giu
nguyen ty le va letterbox thanh dung `640x640` (mau nen RGB 114). Khong keo
gian crop theo hai chieu nen hinh dang vat the khong bi meo.

Nhan giu `A` hoac `D` de di chuyen lien tuc qua anh truoc/sau; tha phim de
dung ngay.

Anh bi xoa duoc di chuyen vao `work/archive/removed` thay vi xoa vinh vien.
Dieu nay giu toc do thao tac nhanh nhung van cho phep `Ctrl+Z`.

## Data phan loai don gian

Chon che do `Data phan loai don gian (thu muc class)` va mo thu muc `data`.
Ten cac thu muc con truc tiep duoc dung lam class, khong con phu thuoc nam
class xoai co dinh:

```text
data\
  class_A\*.jpg
  class_B\*.jpg
  class_C\*.jpg
```

Tool tu sap xep ten class, quet ca anh trong thu muc long ben trong moi class
va hien so anh ngay tren tung nut. Class 1 den 9 co phim tat; class thu 10 tro
di chon bang click. Bo loc class, thanh keo tien do, zoom/pan va nhan giu
`A`/`D` van hoat dong nhu cac editor khac.

- `Enter`: di chuyen anh sang thu muc class dang chon. Duong dan con duoc giu
  lai; neu trung ten, tool tao hau to `__classeditNNN` va khong ghi de.
- `F`: bo lua chon class chua luu va tro ve class thu muc ban dau.
- `Delete`: loai anh khoi thu muc `data`.
- `Ctrl+Z`: bo thay doi chua luu, hoac phuc hoi lan di chuyen/xoa gan nhat.

### Luu thanh dataset moi, khong sua dataset nguon

Nut `Sua truc tiep dataset goc` trong muc `CHE DO LUU` co the bat thanh
`Luu thanh dataset moi`. Khi bat, tool yeu cau chon mot thu muc cha va de xuat
ten `<ten_data>_edited`; duong dan day du van co the sua tay neu can.

- `Enter` chi xep doi class trong bo nho va chuyen sang anh tiep theo.
- `Delete` chi danh dau anh se bi loai khoi ban dataset moi.
- `Ctrl+Z` hoan tac cac thao tac staging theo thu tu LIFO.
- Bo dem class, bo loc class va danh sach anh cap nhat theo trang thai staging,
  nhung moi file trong dataset nguon van giu nguyen duong dan va noi dung.
- Nut `XAC NHAN Tao dataset moi` moi thuc su sao chep anh sang dich. Dialog tien
  do cho biet anh dang xu ly va cho phep `Huy an toan`.

Thu muc dich phai chua ton tai hoac dang rong, nam ngoai dataset nguon. Tool
tao tat ca thu muc class, giu duong dan long khi co the va them hau to
`__classeditNNN` neu hai anh roi vao cung duong dan. Ket qua duoc dung trong
thu muc tam cung volume va chi publish bang atomic rename sau khi sao chep
thanh cong toan bo. Loi doc/ghi, file nguon bi thay doi, dich xuat hien giua
chung hoac huy thao tac deu xoa ban tam; dataset nguon va noi dung co san tai
dich khong bi ghi de.

Neu con staging chua xuat, tool hoi xac nhan truoc khi tat che do, doi mode,
mo dataset khac hay dong ung dung. Sau khi xuat, duong dan dich duoc xoa khoi
o nhap de lan xuat tiep theo khong vo tinh ghi vao bo vua tao.

Mode nay khong tao hay sua `data.yaml`, `manifest.csv`, split hoac bbox. Anh
bi `Delete` duoc dua ra archive nam canh dataset:

```text
<thu_muc_cha>\.cvat_nhai_simple_archive\<dataset_id>\
```

Archive nam ngoai `data`, nen khong bi framework ImageFolder nhan nham la mot
class. Tool tu choi anh nam truc tiep o goc `data`, symlink/junction va layout
YOLO/class_f de tranh mo nham che do. Moi thao tac ghi journal, chay ngoai UI
thread va rollback duong dan cung bo dem neu ghi journal that bai.

## Split

`Tu dong` uu tien ten thu muc `train`, `val`, `test` neu no nam trong duong
dan anh nguon. Neu khong co, split duoc chon on dinh theo hash duong dan voi
ty le 70/20/10. Co the ep split bang combobox tren giao dien.

## Kiem tra CLI

Dry-run migration, khong thay doi du lieu:

```powershell
.\.venv\Scripts\cvat-nhai-cli.exe migrate
```

Audit YAML:

```powershell
.\.venv\Scripts\cvat-nhai-cli.exe audit
```

Chay test:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Nguyen tac an toan

- Khong xoa anh nguon neu mot output hoac metadata ghi that bai.
- Khong cho quet/gan nhan truc tiep anh nam trong hai dataset dich.
- Ten file trung duoc them hash, khong ghi de.
- Ghi YAML/JSON theo kieu atomic replace.
- Ghi journal JSONL cho moi thao tac thanh cong va that bai.
- Migration 4 sang 5 lop luon co backup va se dung neu manifest khong khop.
