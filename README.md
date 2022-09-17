

# EC4SRec (Coming Soon)
Code for the CIKM2022 Paper "Explanation Guided Contrastive Learning for Sequential Recommendation"

## EC4SRec News

**09/17/2022**: We updated SASRec and CL4SRec.

## EC4SRec To-Do list

- [x] Update SASRec.
- [x] Update CL4SRec.
- [ ] Update CL4SRec w/ X-Aug (Ours).
- [ ] Update CoSeRec.
- [ ] Update CoSeRec w/ X-Aug (Ours).
- [ ] Update ContraRec.
- [ ] Update ContraRec w/ X-Aug (Ours).
- [ ] Update DuoRec.
- [ ] Update DuoRec w/ X-Aug (Ours).
- [ ] Update ICL.
- [ ] Update EC4SRec (Ours).

## Quick Start

Command for SASRec on ml-100k
~~~
python run_train.py --dataset=ml-100k --model=SASRec --method=None --gpu_id=1 
~~~

Command for CL4SRec over SASRec on ml-100k
~~~
python run_train.py --dataset=ml-100k --model=SASRec --method=CL4SRec --gpu_id=0 
~~~

## Performance

#### Experimtents on ML-100K
| model            | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|------------------|----------|-----------|-----------|--------|---------|---------|
| SASRec           |0.0700    |0.1304     |0.2269     |0.0423  |0.0617   |0.0860   |
| CL4SRec          |0.0753    |0.1273     |0.2216     |0.0452  |0.0617   |0.0856   |
|  w/ X-Aug        |          |           |           |        |         |         |
| CoSeRec          |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| ContraRec        |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| DuoRec           |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| ICL              |          |           |           |        |         |         |
| EC4SRec          |          |           |           |        |         |         |



#### Experimtents on Amazon_Beauty
| model            | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|------------------|----------|-----------|-----------|--------|---------|---------|
| SASRec           |          |           |           |        |         |         |
| CL4SRec          |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| CoSeRec          |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| ContraRec        |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| DuoRec           |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| ICL              |          |           |           |        |         |         |
| EC4SRec          |          |           |           |        |         |         |



#### Experimtents on Amazon_Clothing_Shoes_and_Jewelry
| model            | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|------------------|----------|-----------|-----------|--------|---------|---------|
| SASRec           |          |           |           |        |         |         |
| CL4SRec          |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| CoSeRec          |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| ContraRec        |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| DuoRec           |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| ICL              |          |           |           |        |         |         |
| EC4SRec          |          |           |           |        |         |         |



#### Experimtents on Amazon_Sports_and_Outdoors
| model            | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|------------------|----------|-----------|-----------|--------|---------|---------|
| SASRec           |          |           |           |        |         |         |
| CL4SRec          |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| CoSeRec          |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| ContraRec        |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| DuoRec           |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| ICL              |          |           |           |        |         |         |
| EC4SRec          |          |           |           |        |         |         |



#### Experimtents on ML-10M
| model            | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|------------------|----------|-----------|-----------|--------|---------|---------|
| SASRec           |          |           |           |        |         |         |
| CL4SRec          |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| CoSeRec          |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| ContraRec        |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| DuoRec           |          |           |           |        |         |         |
|  w/ X-Aug        |          |           |           |        |         |         |
| ICL              |          |           |           |        |         |         |
| EC4SRec          |          |           |           |        |         |         |