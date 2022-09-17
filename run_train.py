# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn


import argparse

from recbole.quick_start import run_recbole


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SASRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')
    parser.add_argument('--method', type=str, default='CL4SRec', help='None, CL4SRec, ...')
    parser.add_argument('--cl_loss_weight', type=float, default=0.1, help='weight for contrastive loss')
    parser.add_argument('--temp_ratio', type=float, default=1.0, help='temperature ratio')

    parser.add_argument('--gpu_id', type=int, default=1, help='temperature ratio')

    ### ours
    parser.add_argument('--xai_method', type=str, default='occlusion', help='saliency, ig, occlusion')
    parser.add_argument('--xai_update_freq', type=int, default=3, help='2,4,6,8,10')

    args, _ = parser.parse_known_args()


    config_dict = {
        'neg_sampling': None,
        'method': args.method,
        'cl_loss_weight': args.cl_loss_weight,
        'temp_ratio': args.temp_ratio,
        'gpu_id': args.gpu_id,

        'xai_method': args.xai_method,
        'xai_update_freq': args.xai_update_freq,
    }

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=config_dict)
