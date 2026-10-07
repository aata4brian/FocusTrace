# Operator Guide

## Before the participant arrives

1. Use Python 3.11/3.12 and install dependencies.
2. Put laptop, operator phone, and stimulus device on the same private Wi-Fi/hotspot.
3. Run `python scripts/webcam_diagnostic.py` and `python scripts/network_diagnostic.py`.
4. Confirm local A2 and B2 video assets and C1 feed assets are present.
5. Start `python run.py` and note the printed operator PIN and URLs.

## Device roles

- `/operator`: researcher controls setup, START/NEXT/STOP, review, and monitoring.
- `/participant`: participant sees only assigned task content and sends task responses.
- `/stimulus`: external task-relevant or task-irrelevant standardized content for B2/C1.
- `/monitor`: researcher-only live monitoring page.

Participant/stimulus clients do not receive operator privileges. All authoritative timestamps are created on the laptop server, not the phone clock.

## Start a session

1. Pair the operator page using the printed PIN.
2. Open participant/stimulus pages and confirm they are connected.
3. Enter an anonymous participant ID (for example `P01`).
4. Select the preset for that ID or a valid custom sequence containing A1–C2 exactly once.
5. Confirm informed consent has been obtained according to the approved protocol.
6. Run preflight. Do not start with a critical failure.
7. Press START. Calibration runs for 30 seconds while continuous raw video recording begins.
8. At each `TRANSITION`, set up the next condition and press NEXT only when ready.
9. After C2 completes, press STOP to finalize files and the integrity report.

## After the session

1. Open the session review page.
2. Check integrity status, completed block order, missing responses, device disconnects, and recording gaps.
3. Review uncertain time ranges against raw video before adding verified corrections.
4. Never overwrite raw/initial labels; corrections must remain auditable.
5. Back up de-identified research outputs according to the study data-management plan.

## Abort policy

If consent is withdrawn, camera/storage fails, or the experiment can no longer be run safely/reliably, use ABORT rather than improvising labels. An aborted session remains recorded in metadata and is excluded from normal dataset preparation.

## Instruction gate pada halaman Operator

Halaman `/operator` menampilkan panduan kontekstual yang harus diselesaikan sebelum tombol kontrol utama digunakan.

Sebelum `START SESSION`, peneliti menyelesaikan berurutan: Penyambutan dan Persiapan, Pengaturan Posisi Duduk, Penjelasan Aktivitas, lalu Instruksi Personalized Calibration. Setelah empat tahap selesai, `START SESSION` dapat digunakan sesuai status pre-flight dan consent.

Setiap kali sistem masuk `TRANSITION`, panduan pertama menampilkan naskah transisi. Setelah itu panduan menampilkan instruksi untuk `next_block` aktual sesuai sequence peserta (misalnya B1 atau C2). Tombol `Mulai blok berikutnya` baru boleh digunakan setelah peneliti menandai bahwa instruksi blok tersebut sudah dibacakan.

Setelah blok keenam, panduan meminta konfirmasi sebelum `STOP SESSION`. Sesudah sesi berstatus `FINISHED`, panduan melanjutkan Penyelesaian Eksperimen, Debriefing, dan Penutupan secara berurutan.
