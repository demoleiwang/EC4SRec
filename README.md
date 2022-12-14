

# EC4SRec (Updating)
Code for the CIKM2022 Paper "Explanation Guided Contrastive Learning for Sequential Recommendation"

![visitors](https://visitor-badge.glitch.me/badge?page_id=demoleiwang/EC4SRec)

## EC4SRec News

**09/27/2022**: We updated DuoRec on ml-100k.<br/>
**09/19/2022**: We updated CL4SRec w/ X-Aug (Ours) on ml-100k.<br/>
**09/17/2022**: We updated SASRec and CL4SRec on ml-100k.

## EC4SRec To-Do list

- [x] Update SASRec.
- [x] Update CL4SRec.
- [x] Update CL4SRec w/ X-Aug (Ours).
- [ ] Update CoSeRec.
- [ ] Update CoSeRec w/ X-Aug (Ours).
- [ ] Update ContraRec.
- [ ] Update ContraRec w/ X-Aug (Ours).
- [x] Update DuoRec.
- [x] Update DuoRec w/ X-Aug (Ours).
- [ ] Update ICL.
- [ ] Update EC4SRec (Ours).

## Quick Start

Command for CL4SRec 
~~~
python run_train.py --dataset=ml-100k --model=SASRec --method=CL4SRec --gpu_id=13
python run_train.py --dataset=Amazon_Beauty --model=SASRec --method=CL4SRec --gpu_id=4
~~~
Command for CL4SRec w/ X-Aug (Ours)
~~~
python run_train.py --dataset=ml-100k --model=SASRec --method=CL4SRec_XAUG --xai_method=occlusion --gpu_id=2
python run_train.py --dataset=Amazon_Beauty --model=SASRec --method=CL4SRec_XAUG --xai_method=occlusion --gpu_id=12
~~~
Command for DuoRec
~~~
python run_train.py --dataset=ml-100k --model=SASRec --method=DuoRec --gpu_id=2
python run_train.py --dataset=Amazon_Beauty --model=SASRec --method=DuoRec --gpu_id=12
~~~
Command for DuoRec w/ X-Aug (Ours)
~~~
python run_train.py --dataset=ml-100k --model=SASRec --method=DuoRec_XAUG --xai_method=occlusion --gpu_id=2
python run_train.py --dataset=Amazon_Beauty --model=SASRec --method=DuoRec_XAUG --xai_method=occlusion --gpu_id=12
~~~

## Performance

### 5-core Setting

#### Experimtents on ML-100K.
| model            | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|------------------|----------|-----------|-----------|--------|---------|---------|
| CL4SRec          |0.0753    |0.1273     |0.2216     |0.0452  |0.0617   |0.0856   |
|+X-Aug (Occlusion)|0.0806    |0.1389     |0.2418     |0.0454  |0.0642   |0.0899   |
| CoSeRec          |          |           |           |        |         |         |
|+X-Aug (Occlusion)|          |           |           |        |         |         |
| ContraRec        |          |           |           |        |         |         |
|+X-Aug (Occlusion)|          |           |           |        |         |         |
| DuoRec           |0.0764    |0.1198     |0.2216     |0.0462  |0.0602   |0.0860   |
|+X-Aug (Occlusion)|0.0753    |0.1315     |0.2238     |0.0456  |0.0633   |0.0866   |
| ICL              |          |           |           |        |         |         |
|EC4SRec (Occlusion)|         |           |           |        |         |         |




#### Experimtents on Amazon_Beauty
| model            | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|------------------|----------|-----------|-----------|--------|---------|---------|
| CL4SRec          |          |           |           |        |         |         |
|+X-Aug (Occlusion)|          |           |           |        |         |         |
| CoSeRec          |          |           |           |        |         |         |
|+X-Aug (Occlusion)|          |           |           |        |         |         |
| ContraRec        |          |           |           |        |         |         |
|+X-Aug (Occlusion)|          |           |           |        |         |         |
| DuoRec           |          |           |           |        |         |         |
|+X-Aug (Occlusion)|          |           |           |        |         |         |
| ICL              |          |           |           |        |         |         |
|EC4SRec (Occlusion)|         |           |           |        |         |         |


## Citations

```bibtex
@article{wang2022explanation,
  title={Explanation Guided Contrastive Learning for Sequential Recommendation},
  author={Wang, Lei and Lim, Ee-Peng and Liu, Zhiwei and Zhao, Tianxiang},
  journal={arXiv preprint arXiv:2209.01347},
  year={2022}
}
```
