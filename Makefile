PY := .venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

.PHONY: data corpus sft-data pretrain domain sft rag chat serve eval bench

data:
	$(PY) scripts/scrape_data.py
	$(PY) scripts/scrape_subarchivist.py
	$(PY) scripts/scrape_acdb.py --seasons 4 --max-pages 40

corpus:
	$(PY) scripts/build_corpus_v2.py
	$(PY) scripts/gen_persona.py --per-line 2
	$(PY) scripts/filter_pairs.py data/processed/synth_persona_v2.jsonl
	$(PY) scripts/gen_identity_qa.py
	$(PY) scripts/gen_entity_qa.py
	$(PY) scripts/paraphrase_questions.py

sft-data:
	$(PY) scripts/build_sft.py --config configs/sft_sources_clean.json
	$(PY) scripts/prepare_data_v2.py

pretrain:
	$(PY) scripts/train.py --config configs/pretrain_general_56m.json --out checkpoints/pretrain_56m

domain:
	$(PY) scripts/train.py --config configs/pretrain_domain_56m.json --out checkpoints/domain_56m

sft:
	$(PY) scripts/sft.py --config configs/sft.json --ckpt checkpoints/domain_56m/best.pt \
		--out checkpoints/sft_56m --epochs 2 --lr 1e-4 --dropout 0.1

rag:
	$(PY) scripts/build_rag_index.py

chat:
	$(PY) scripts/chat.py --ckpt checkpoints/sft_56m/best.pt

serve:
	$(PY) scripts/serve.py --ckpt checkpoints/sft_56m/best.pt --port 8000

eval:
	$(PY) scripts/eval.py --ckpt checkpoints/sft_56m/best.pt --base-ckpt checkpoints/domain_56m/best.pt \
		--judge Qwen/Qwen2.5-1.5B-Instruct --rag-index data/processed/rag_index.npz \
		--out reports/eval.json

bench:
	$(PY) scripts/benchmark.py --ckpt checkpoints/sft_56m/best.pt --cpu --out reports/benchmark.json
