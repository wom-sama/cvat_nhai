# Bao cao kiem dinh dataset editor - 2026-07-30

Phien ban sau kiem dinh: `1.5.1`.

## Pham vi va nguyen tac an toan

- Dataset nguon chi duoc doc:
  - `D:\DataAI\AIEx\newdataset\class_f`
  - `D:\DataAI\AIEx\newdataset\yolo_f`
- Hai ban sao vat ly duoc tao bang Robocopy (`/E /COPY:DAT /DCOPY:DAT /XJ`), khong dung hardlink:
  - `D:\DataAI\AIEx\CVAT_\work\dataset_editor_audit_20260730\class_f_copy`
  - `D:\DataAI\AIEx\CVAT_\work\dataset_editor_audit_20260730\yolo_f_copy`
- Tat ca thao tac sua class, sua bbox, xoa va hoan tac chi chay tren hai ban sao tren.
- Thu muc audit da duoc xoa sau khi hoan tat. Khong giu lai anh, label hay archive thu nghiem.

## Loi phat hien va cach sua

### 1. Manifest khong hoat dong khi dataset duoc copy hoac di chuyen

**Hien tuong:** `manifest.csv` cua ca hai dataset luu duong dan tuyet doi tro ve thu muc cu. Khi doi class tren ban sao `class_f`, tool khong tim thay dong cu va chen them dong moi. Test thuc te lam manifest tang sai tu 12.849 len 12.850 dong.

**Sua:** bo sung chi muc duong dan co kha nang rebase theo `split`, class va phan duong dan tuong doi cua anh. Duong dan cu van duoc bao toan de Ctrl+Z khoi phuc chinh xac, nhung thao tac tren dataset da di chuyen se cap nhat dung dong thay vi tao dong trung.

### 2. Xoa anh YOLO de lai dong manifest mo coi

**Hien tuong:** xoa anh va label thanh cong nhung `manifest.csv` van con dong tro den file da xoa. Tren ban sao that, manifest van giu 11.908 dong sau khi xoa mot mau.

**Sua:** xoa tat ca dong manifest khop voi anh trong cung giao dich voi image, label va `canbang.yaml`. Ctrl+Z chen lai dung dong, dung vi tri va dung thu tu ban dau, ke ca manifest co cac dong trung lap.

### 3. Giao dich YOLO co the khong dong bo khi ghi metadata hoac journal loi

**Hien tuong:** journal commit truoc day nam ngoai khoi phuc giao dich; neu ghi journal loi sau khi file da thay doi thi thao tac khong co ban ghi de Ctrl+Z. Cach rollback balance bang phep cap nhat nguoc cung co the that bai lan hai.

**Sua:** snapshot byte cua label, `canbang.yaml` va manifest truoc moi giao dich. Save, delete va undo chi duoc xem la thanh cong sau khi journal commit. Bat ky loi ghi manifest, balance hoac journal nao cung khoi phuc dung byte va vi tri file truoc do.

### 4. Truong hop manifest co hai dong giong het nhau

**Hien tuong:** test bien moi phat hien lan hoan tac dau tien tu choi dong thu hai vi noi dung giong dong thu nhat.

**Sua:** khoi phuc theo index va thu tu goc cua tung dong thay vi coi noi dung trung lap la xung dot.

### 5. Test UI co the doc label trong luc worker dang thay file

**Hien tuong:** mot vong test cuoi tai hien `PermissionError` tren Windows vi ham cho doc label truoc khi task undo bat dong bo ket thuc. Day la race trong test harness; giao dien dang busy va khong cho nguoi dung thao tac trong khoang nay.

**Sua:** dieu kien test doi `active_tasks` ve rong truoc, sau do moi doc noi dung label. Thu tu nay kiem tra dung hop dong cua giao dien va loai bo truy cap file giua giao dich.

## Kiem dinh toan bo ban sao

### class_f

- 12.849 anh va 12.849 dong manifest.
- 0 anh hong; 0 anh sai kich thuoc 640x640.
- 0 duong dan manifest mat sau khi rebase; 0 target trung; 0 sai class/folder.
- `canbang.yaml` va `stats.json` khop tat ca class trong train/val/test.
- 0 nhom anh trung SHA-256 xuyen split; 0 anh giong nhau nhung nam o hai class.

### yolo_f

- 11.908 anh, 11.908 label va 13.088 bbox.
- 0 anh hong; 0 label thieu; 0 label mo coi; 0 label rong.
- 0 loi parse, class ngoai schema, bbox zero-size hoac bbox vuot bien.
- 0 duong dan manifest mat sau khi rebase; 0 target trung.
- `canbang.yaml` khop so anh va object cua tung class trong tung split.
- 0 `leakage_group` xuyen split; 0 nhom anh trung SHA-256 xuyen split.

### Sai lech co san giua hai bo

- `yolo_f` co 13.088 bbox, trong khi `class_f` co 12.849 crop: `class_f` it hon 239 object.
- 78 anh YOLO khong con crop nao; tong cong 239 chi so object khong con trong `class_f`.
- 439 object cung `source_image` va cung index dang co class khac nhau giua hai bo.
- Khong co object thua trong `class_f`, khong co sai split va khong co source chi ton tai o `class_f`.

Day la trang thai da ton tai truoc kiem thu, phu hop voi viec hai che do da duoc chinh rieng. Audit khong tu dong dong bo vi lam vay co the ghi de quyet dinh gan nhan cua nguoi dung. Khi can hai bo dong nhat, hay xuat lai `yolo_f + class_f` tu bo YOLO da chot.

## Thu nghiem thao tac that tren ban sao

- `class_f`: 5 thao tac lien tiep gom doi class hai lan, xoa anh da doi, doi class o val va xoa anh test; sau do undo LIFO 5 lan.
- YOLO: sua class tren anh nhieu bbox, sua class va toa do bbox lan hai, xoa anh da sua va xoa them anh khac; sau do undo LIFO 4 lan.
- Sau undo, manifest, balance, anh va label duoc so byte voi snapshot va khop hoan toan.
- Thoi gian ghi tren dataset that:
  - `class_f`: 83,55-114,62 ms/thao tac; undo 96,29-131,61 ms.
  - YOLO edit: 12,97-42,74 ms; YOLO delete 105,21-109,59 ms; undo 26,15-106,76 ms.

## Giao dien va test tu dong

- Mo ban sao YOLO 11.908 anh trong giao dien offscreen: 1,99 giay; anh dau va bbox hien thi dung.
- Mo ban sao `class_f` 12.849 anh: 1,68 giay; anh dau va class hien thi dung.
- Bo test bao phu scan bat dong bo, busy overlay, giu A/D, thanh seek, loc split/class, Tab chuyen bbox, save/delete/undo, export, chong leak, crop/letterbox 640, padding preview, dataset relocation va cac loi ghi cuong buc.
- Ket qua cuoi: 82 test passed; `pip check` khong phat hien dependency hong.

## Chung minh dataset nguon khong doi

So sanh dau va cuoi phien:

| Dataset | File | Byte |
|---|---:|---:|
| class_f | 13.092 | 1.120.472.700 |
| yolo_f | 23.821 | 663.011.269 |

SHA-256 metadata dau va cuoi giong nhau:

- class_f `data.yaml`: `312EB376E89023F42E9DF24FEA8723B64A6B885C15C24F85D8E319D1FF4F1FA8`
- class_f `manifest.csv`: `2AC9142F643E8CB6577499C24E1738A64D7C49BA9FBEB9322A96C6F21DF68A17`
- class_f `canbang.yaml`: `BEA4901868C6C87734D493A6C6135C49653D98BA9364BA6BE4F4B9C9F6843462`
- class_f `stats.json`: `0B343F2DB2A4EC96BDD4430B82C722FE1FC9AE87FA24F4B2A2C4008C13F22B49`
- yolo_f `data.yaml`: `716E33DF24C63A9E9920F97B685199707FB84AB4C7154544F5DD9A3E00D884EF`
- yolo_f `manifest.csv`: `EB16E09CD20FFF8480B50AA79E7D5FEC779E4614398911E2E257BA6A37642FF2`
- yolo_f `canbang.yaml`: `05808224579A5F73A31F8655056A9A4727F9185076019F7175242E374F0B5227`

Kiem tra toan bo SHA-256 cua ca nguon va ban sao cung luc khong duoc dung lam bang chung vi vuot thoi gian gioi han. Bang chung an toan duoc dua tren Robocopy thanh cong, so file/byte, hash metadata, quet/parse toan bo noi dung ban sao va viec gioi han moi duong dan ghi vao thu muc audit.
