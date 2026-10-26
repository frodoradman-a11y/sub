# راهنمای کامل SubASR

این سند همهٔ چیزهایی را که برای آماده‌سازی داده، آموزش، ارزیابی و ساخت زیرنویس لازم است توضیح می‌دهد. طراحی و دلایل انتخاب‌ها در [architecture-proposal.md](architecture-proposal.md) آمده است.

---

## ۱. معماری پیاده‌سازی‌شده

```
صوت 16kHz
  └─► انکودر صوتی Gemma 4 (Conformer به سبک USM، فریم ۴۰ms = ۲۵ فریم در ثانیه)
        ├─► سر TDT  ──────────────────────────► حالت fast (بدون LLM)
        ├─► سر CTC  ──► فرضیهٔ اولیه ──┐
        ├─► سر رویداد صوتی (گفتار/آواز/موسیقی/نویز/خنده)
        └─► Adapter (Window Q-Former یا پروجکتور خود Gemma)
                └─► LLM متنی Gemma 4 + LoRA
                        ├─► حالت edit: یک پاس دوطرفه روی فرضیهٔ CTC (NLE)
                        └─► حالت quality: تولید خودبازگشتی با بافت و واژه‌نامه
متن نهایی ─► هم‌ترازی اجباری CTC ─► زمان هر کلمه ─► ساخت زیرنویس ─► SRT/VTT
```

| بخش | فایل | توضیح |
|---|---|---|
| انکودر Gemma 4 | `subasr/modules/encoders.py` (`Gemma4AudioEncoder`) | برج صوتی Gemma 4 با feature extractor خودش؛ خروجی لایه‌ها با hook گرفته می‌شود |
| انکودر Conformer | `subasr/modules/encoders.py` (`ConformerEncoder`) | انکودر کوچک از صفر (برای تست و آزمایش) |
| سر CTC، TDT، رویداد | `subasr/modules/heads.py` | TDT با مدت‌زمان‌های `[0,1,2,3,4]` |
| تابع هزینهٔ TDT | `subasr/losses/tdt.py` | پیاده‌سازی دقیق در PyTorch خالص؛ با شمارش همهٔ مسیرها تست شده |
| Adapter | `subasr/modules/adapter.py` | `WindowQFormer` (۱۲ توکن در ثانیه) یا `GemmaNativeAdapter` (۲۵ توکن در ثانیه، از قبل هم‌تراز) |
| LLM | `subasr/modules/llm.py` | `Gemma4LM` (ورودی ترکیبی متن و صوت، ماسک دوطرفه، تولید با KV-cache) و `TinyLM` برای تست |
| LoRA | `subasr/modules/lora.py` | پیاده‌سازی مستقل، با امکان ادغام (merge) |
| مدل کامل | `subasr/model.py` | هزینه‌های آموزش، حالت‌های استنتاج، محافظ توهم، زمان‌بندی کلمه‌ها |
| داده | `subasr/data.py`، `subasr/audio/*` | manifest، دسته‌بندی بر اساس طول، افزونگی موسیقی و نویز، قطعه‌بندی فیلم |
| متن فارسی | `subasr/text/normalize.py` | یکسان‌سازی ی/ي، ک/ك، حذف اعراب، نیم‌فاصله، ارقام |
| زیرنویس | `subasr/subtitles.py` | قواعد خط و زمان، برچسب موسیقی، SRT و VTT |
| خط فرمان | `train.py`، `infer.py`، `evaluate.py`، `tools/build_vocab.py` | |

### چهار حالت استنتاج

| حالت | چه اجرا می‌شود | کاربرد |
|---|---|---|
| `fast` | انکودر + TDT | سریع‌ترین؛ بدون LLM |
| `edit` | فرضیهٔ CTC + یک پاس دوطرفهٔ LLM | کیفیت نزدیک به LLM با سرعت زیاد |
| `quality` | تولید خودبازگشتی LLM با prompt (زبان، زیرنویس قبلی، واژه‌نامه) | بیشترین دقت، کندتر |
| `auto` | همه با `fast`؛ قطعه‌های کم‌اطمینان یا دارای موسیقی به `edit` (یا `quality`) می‌روند | پیشنهاد پیش‌فرض برای فیلم |

**محافظ توهم در حالت quality:** اگر خروجی LLM با فرضیهٔ CTC بیش از حد مجاز (پیش‌فرض CER > ۰٫۶) فرق داشته باشد، نتیجهٔ حالت edit جایگزین می‌شود. در خروجی JSON این قطعه‌ها با `quality->edit` علامت می‌خورند.

**حذف متن در بخش‌های بی‌گفتار:** اگر احتمال گفتار یا آواز در کل قطعه کمتر از `--nonspeech-threshold` باشد (پیش‌فرض ۰٫۲)، متنی برای آن قطعه تولید نمی‌شود.

---

## ۲. نصب

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch            # نسخهٔ CUDA مناسب کارت گرافیک خودتان
pip install -e ".[dev]"      # transformers>=5، soundfile، pyyaml، pytest
```

- برای خواندن mp3/mp4/mkv باید **ffmpeg** نصب باشد.
- برای دانلود Gemma 4 باید مجوز مدل را در Hugging Face بپذیرید و `huggingface-cli login` کنید.
- نسخهٔ transformers باید کلاس‌های `Gemma4*` را داشته باشد (کد با نسخهٔ 5.20 تست شده است).

---

## ۳. آماده‌سازی داده

### ۳.۱ قالب manifest
هر خط یک شیء JSON است. فقط `audio` و `text` اجباری‌اند:

```json
{"audio": "clips/0001.wav", "text": "سلام، حالت چطوره؟", "duration": 2.4, "language": "fa"}
{"audio": "films/a.mp4", "offset": 125.3, "duration": 6.1, "text": "...", "labels": ["speech", "music"]}
{"audio": "music/inst_01.wav", "text": "", "labels": ["music"]}
{"audio": "clips/0002.wav", "text": "...", "context": "زیرنویس قبلی", "glossary": ["کیارستمی"]}
```

| فیلد | توضیح |
|---|---|
| `audio` | مسیر فایل (نسبی به پوشهٔ manifest). wav/flac/ogg مستقیم و بقیه با ffmpeg خوانده می‌شوند |
| `text` | متن. **خالی** برای کلیپ‌های بدون گفتار (موسیقی، نویز)؛ این نمونه‌ها به مدل یاد می‌دهند که متن نسازد |
| `duration`، `offset` | برای برش از فایل بلند. اگر `duration` نباشد، از روی فایل خوانده می‌شود (کند) |
| `language` | پیش‌فرض: `train.default_language` |
| `labels` | رویدادهای صوتی کلیپ از بین `speech`، `singing`، `music`، `noise` و `laughter`. اگر نباشد و متن خالی نباشد، `speech` فرض می‌شود |
| `context`، `glossary` | برای آموزش استفاده از بافت و واژه‌نامه در حالت quality |

### ۳.۲ واژگان CTC/TDT

```bash
python -m subasr.tools.build_vocab --manifest data/train.jsonl --output data/vocab.txt --min-count 2
```

واژگان نویسه‌ای است: برای فارسی حدود ۴۰ تا ۸۰ نویسه می‌شود. توکن ۰ «blank» است. برای SentencePiece، `model.ctc_tokenizer.type: spm` را بگذارید و مسیر فایل مدل را بدهید. اما واژگان بزرگ حافظهٔ TDT را زیاد می‌کند (بخش ۹ را ببینید).

### ۳.۳ موسیقی و نویز برای افزونگی
دو manifest با فیلد `audio` بسازید (`data/music.jsonl` و `data/noise.jsonl`). در آموزش، گفتار با احتمال `music_prob` در SNR تصادفی بین ۵- و ۲۰+ دسی‌بل با موسیقی میکس می‌شود و برچسب `music` خودکار روشن می‌شود. هم موسیقی بی‌کلام لازم است و هم آهنگ با کلام. آهنگ با کلام را با متن آن به‌عنوان دادهٔ اصلی و با برچسب `singing` اضافه کنید.

### ۳.۴ نرمال‌سازی فارسی
متن هدف قبل از آموزش با `normalize_text` یکسان می‌شود:
- ي/ى → ی و ك → ک
- حذف اعراب و کشیده
- پاک‌سازی نیم‌فاصله‌های تکراری یا کنار فاصله

برای ارزیابی، `normalize_for_scoring` علاوه بر این نقطه‌گذاری را حذف و ارقام را یکسان می‌کند. نیم‌فاصله به‌طور پیش‌فرض حذف می‌شود («می‌روم» = «میروم»).

---

## ۴. پیکربندی

پیکربندی در YAML است ([configs/gemma4_e2b.yaml](../configs/gemma4_e2b.yaml)). هر کلید را می‌توان از خط فرمان با نقطه تغییر داد:

```bash
python -m subasr.train --config configs/gemma4_e2b.yaml train.stage=joint train.lr=3e-4 model.llm.lora.r=32
```

مهم‌ترین کلیدها:

| کلید | پیش‌فرض (gemma4_e2b) | توضیح |
|---|---|---|
| `model.encoder.type` | `gemma4` | `gemma4` یا `conformer` |
| `model.encoder.pretrained` | `google/gemma-4-E2B-it` | شناسهٔ HF یا پوشهٔ محلی |
| `model.encoder.adapter_layers` | `[4, 8, -1]` | لایه‌هایی از انکودر که Q-Former می‌خواند |
| `model.heads.tdt_durations` | `[0,1,2,3,4]` | باید ۱ را داشته باشد |
| `model.adapter.type` | `window_qformer` | یا `gemma_native` |
| `model.adapter.num_queries` / `window_frames` | ۱۲ / ۲۵ | تعداد توکن ورودی LLM در هر ثانیه = num_queries × 25 / window_frames |
| `model.llm.dtype` | `bfloat16` | وزن‌های منجمد LLM. انکودر و Adapter همیشه fp32 نگه داشته می‌شوند |
| `model.llm.lora.r` / `alpha` | ۶۴ / ۱۲۸ | |
| `model.llm.prompt_template` | `chat` | قالب چت خود توکنایزر Gemma. یا رشته‌ای با `{audio}` و `{instruction}` |
| `model.llm.instruction` | … | `{language}` با نام زبان جایگزین می‌شود |
| `model.llm.nle_slots_per_token` | ۱ | تعداد جای خالی بعد از هر توکن فرضیه در حالت edit (ظرفیت درج) |
| `train.stage` | `encoder` | `encoder`، `align` یا `joint` |
| `train.batch_seconds` | ۶۰۰ | مجموع ثانیه‌های صوت در هر batch (با padding) |
| `train.lr` / `encoder_lr` | 5e-4 / 5e-5 | نرخ یادگیری سرها، Adapter و LoRA / نرخ یادگیری انکودر |
| `train.loss.*` | ctc 0.3، tdt 1، events 0.1، ar 1، nle 0.5 | وزن هزینه‌ها |
| `train.nle_corrupt_prob` | ۰٫۵ | سهم فرضیه‌های ساختگی (خراب‌شدهٔ متن مرجع) در آموزش حالت edit؛ بقیه از خروجی CTC خود مدل |

---

## ۵. آموزش سه‌مرحله‌ای

| مرحله | چه چیزی آموزش می‌بیند | هزینه‌ها | LLM بارگذاری می‌شود؟ |
|---|---|---|---|
| `encoder` | انکودر Gemma + سرهای CTC/TDT/رویداد | ctc، tdt، events | خیر |
| `align` | فقط Adapter | ar، nle | بله (منجمد) |
| `joint` | انکودر (`encoder_lr`) + سرها + Adapter + LoRA | همه | بله (پایه منجمد) |

```bash
CFG=configs/gemma4_e2b.yaml
python -m subasr.train --config $CFG train.stage=encoder train.output_dir=runs/enc train.max_steps=60000
python -m subasr.train --config $CFG train.stage=align   train.output_dir=runs/align train.init_from=runs/enc/last.pt train.max_steps=5000
python -m subasr.train --config $CFG train.stage=joint   train.output_dir=runs/joint train.init_from=runs/align/last.pt train.max_steps=40000
```

- **خروجی:** گزارش‌ها در `runs/*/log.jsonl` ثبت می‌شوند. هر `eval_every` گام، WER و CER حالت‌ها روی `val_manifest` محاسبه می‌شود.
- **چک‌پوینت:** `last.pt` شامل پیکربندی، واژگان و همهٔ وزن‌های آموزش‌دیده است، **به‌جز وزن‌های منجمد LLM**. آن وزن‌ها هنگام بارگذاری از `model.llm.pretrained` خوانده می‌شوند، پس چک‌پوینت سبک است.
- **مسیر سریع‌تر برای شروع:** `model.adapter.type=gemma_native` پروجکتور از پیش هم‌تراز Gemma را به کار می‌برد. با این گزینه می‌توانید مرحلهٔ `align` را کوتاه کنید یا کنار بگذارید. هزینه‌اش این است که LLM ۲۵ توکن صوتی در ثانیه می‌خواند، نه ۱۲.
- **حافظه (تخمین، اندازه‌گیری نشده):**
  - مرحلهٔ `joint` با Gemma 4 E2B در bf16، انکودر fp32 با AdamW و LoRA با r=64 روی کارتی با حدود ۴۰ تا ۸۰ گیگابایت با `batch_seconds≈600` و gradient checkpointing جا می‌شود.
  - روی کارت ۲۴ گیگابایتی `batch_seconds` را ۱۵۰ تا ۲۵۰ بگذارید و `train.grad_accum` را بالا ببرید.
- **چند GPU:** حلقهٔ آموزش فعلاً تک‌GPU است (بخش ۹).

---

## ۶. ارزیابی

```bash
python -m subasr.evaluate --ckpt runs/joint/last.pt --manifest data/test.jsonl --modes fast edit quality --output preds.jsonl
```

خروجی برای هر حالت WER، CER و RTFx (ثانیهٔ صوت پردازش‌شده در هر ثانیه) را گزارش می‌دهد. `preds.jsonl` متن مرجع، خروجی و حالتی را که واقعاً استفاده شده برای هر جمله نگه می‌دارد.

برای مقایسهٔ منصفانه با Whisper، خروجی Whisper را با همان `subasr.metrics.score` بسنجید تا نرمال‌سازی یکسان باشد.

---

## ۷. ساخت زیرنویس

```bash
python -m subasr.infer --ckpt runs/joint/last.pt --input film.mp4 --output film.fa.srt \
    --mode auto --language fa --glossary names.txt --json segments.json
```

| گزینه | توضیح |
|---|---|
| `--mode` | `fast`، `edit`، `quality` یا `auto` |
| `--auto-escalate` | در حالت auto، قطعه‌های سخت به `edit` (پیش‌فرض) یا `quality` بروند |
| `--glossary` | فایل متنی، هر خط یک اسم یا اصطلاح. در prompt حالت‌های LLM قرار می‌گیرد |
| `--context` | متن قطعهٔ قبلی را به LLM می‌دهد. دقت بهتر، ولی قطعه‌ها یکی‌یکی پردازش می‌شوند |
| `--max-segment` | بیشترین طول قطعه (ثانیه). برش در ساکت‌ترین نقطه نزدیک این حد انجام می‌شود |
| `--max-chars` | بیشترین نویسه در هر خط (پیش‌فرض ۴۲). حداکثر دو خط و مدت ۱ تا ۷ ثانیه |
| `--rtl-embedding` | نویسه‌های RLE/PDF دور هر خط، برای پخش‌کننده‌هایی که راست‌به‌چپ را بد نشان می‌دهند |
| `--nonspeech-threshold` | آستانهٔ حذف متن در قطعه‌های بی‌گفتار؛ عدد منفی یعنی غیرفعال |

بخش‌های موسیقی بدون کلام (که دست‌کم ۱٫۵ ثانیه طول بکشند) با `♪ [موسیقی] ♪` علامت می‌خورند.

استفاده از داخل پایتون:

```python
from subasr.builder import load_checkpoint
from subasr.infer import transcribe_file
from subasr.subtitles import write_subtitles

model, cfg, _ = load_checkpoint("runs/joint/last.pt", "cuda")
cues, segments, words = transcribe_file(model, "film.mp4", mode="auto", glossary=["تهران"])
write_subtitles(cues, "film.srt")
```

---

## ۸. تست‌ها

```bash
python -m pytest -q                 # همه (حدود ۳ دقیقه روی CPU)
python -m pytest -q -m "not slow"   # بدون آموزش سرتاسری (کمتر از یک دقیقه)
```

| فایل | چه چیزی را بررسی می‌کند |
|---|---|
| `test_text.py` | نرمال‌سازی فارسی، WER/CER، توکنایزرها |
| `test_losses_decoding.py` | **هزینهٔ TDT در برابر شمارش کامل همهٔ مسیرها**، گرادیان، CTC greedy، هم‌ترازی اجباری در برابر جست‌وجوی کامل، زمان‌بندی TDT |
| `test_audio.py` | بازنمونه‌برداری، خواندن wav/mp3، میکس با SNR دقیق، افزونگی، قطعه‌بندی فیلم، ویژگی‌ها |
| `test_modules.py` | انکودر، Q-Former (استقلال از padding)، ماسک علّی و دوطرفه، LoRA، هزینهٔ رویداد |
| `test_model.py` | هر سه مرحلهٔ آموزش (مرحلهٔ align فقط Adapter را تغییر می‌دهد)، همهٔ حالت‌ها، زمان‌بندی، چک‌پوینت |
| `test_gemma4.py` | **کلاس‌های واقعی Gemma 4** با وزن تصادفی کوچک: یک گام آموزش با هر دو Adapter، اثر واقعی ماسک دوطرفه، برابری تولید با KV-cache با محاسبهٔ کامل، left padding، چک‌پوینت بدون وزن منجمد |
| `test_end_to_end.py` | آموزش واقعی مدل کوچک روی یک «زبان صوتی» مصنوعی (هر حرف یک بسامد)، سپس ارزیابی هر سه حالت، رونویسی یک فایل بلند با سکوت (متن و زمان شروع جمله‌ها)، و اجرای CLI و ساخت SRT |

نکته: تست Gemma 4 با وزن‌های تصادفی کوچک اجرا می‌شود و چیزی دانلود نمی‌کند. درستی اتصال کد به کلاس‌های واقعی Gemma 4 را نشان می‌دهد، نه کیفیت مدل آموزش‌دیده را.

---

## ۹. محدودیت‌ها و کارهای باقی‌مانده

این موارد در سند طراحی آمده‌اند ولی هنوز پیاده نشده‌اند، یا ساده‌تر از طرح پیاده شده‌اند:

1. **نرخ فریم انکودر ۲۵Hz است، نه ۱۲٫۵Hz.** با تصمیم «فعلاً انکودر خود Gemma» نرخ انکودر همان ۴۰ms است. نرخ ورودی LLM با Q-Former به ۱۲ توکن در ثانیه کاهش یافته است.
2. **مرحلهٔ RL (GRPO/MWER)** و **تقطیر به نسخهٔ Lite** پیاده نشده‌اند.
3. **Aligner غیرخودبازگشتی LLM-محور** پیاده نشده است. به‌جای آن، زمان‌بندی همهٔ حالت‌ها با هم‌ترازی اجباری CTC (Viterbi) انجام می‌شود که ساده‌تر است و مدل جدا لازم ندارد.
4. **تقسیم‌بندی زیرنویس** قاعده‌محور است (طول خط، مکث، پایان جمله). نسخهٔ یادگرفته با توکن‌های `<br>` و `<cue>` پیاده نشده است.
5. **جداسازی صدای خواننده** (BS-/Mel-RoFormer) پیاده نشده است. مقاومت در برابر موسیقی فعلاً از افزونگی داده و سر رویداد صوتی می‌آید.
6. **هزینهٔ TDT** در PyTorch خالص است: دقیق و تست‌شده، ولی حافظهٔ آن متناسب با B×T×U×V است. با واژگان نویسه‌ای فارسی مشکلی نیست. برای واژگان بزرگ (SentencePiece چندهزارتایی) یا قطعه‌های بلند، کرنل CUDA (مثلاً از NeMo) مناسب‌تر است.
7. **آموزش چند GPU** (DDP/FSDP) هنوز اضافه نشده است.
8. حالت edit و quality **یک مجموعه LoRA مشترک** دارند (آموزش چندوظیفه‌ای)، نه دو LoRA جدا.
9. هیچ عدد کیفیتی روی دادهٔ واقعی فارسی گزارش نشده است. کد آماده است، ولی آموزش واقعی به داده و GPU نیاز دارد.
