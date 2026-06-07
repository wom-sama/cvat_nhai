# CVAT Nhai

Ung dung desktop gan lai nhan anh nhanh cho bo du lieu xoai 5 lop. Moi lan
nhan `Enter`, tool tao dong thoi:

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

Nut `Xuat lai YOLO + classification` chon mot thu muc goc rong va tao:

```text
<export>\dataset\images\<split>\*
<export>\dataset\labels\<split>\*.txt
<export>\cls_crops\<split>\<class_name>\*_boxNNN.jpg
```

Ca hai bo dung cung mot split moi theo ty le 70/20/10. Thuat toan can bang
so object cua tung class thay vi chi chia tong so anh. Tat ca bbox cua cung
mot anh, anh co cung family name va anh trung/noi dung gan trung theo visual
fingerprint duoc khoa trong cung mot `leakage_group`, nen khong the nam dong
thoi o train va val/test. `manifest.csv` luu split nguon, split moi va group
de audit data leak.

Moi bo co `data.yaml`, `manifest.csv` va `canbang.yaml`; `cls_crops` co them
`stats.json`. Thu muc export bat buoc phai rong; ket qua duoc dung trong thu
muc tam va chi duoc dua vao dich sau khi toan bo anh, nhan va crop thanh
cong. Thu muc dich phai nam ngoai dataset nguon. Dataset YOLO nguon khong
bi di chuyen hay sap xep lai.

Trong che do sua, `Tab` chon box ke tiep va `Shift+Tab` chon box truoc do,
ke ca khi focus dang nam tren sidebar.

Anh bi xoa duoc di chuyen vao `work/archive/removed` thay vi xoa vinh vien.
Dieu nay giu toc do thao tac nhanh nhung van cho phep `Ctrl+Z`.

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
