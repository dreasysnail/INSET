import os

END_OF_TURN_TOKEN = '<|endofturn|>'
END_OF_TEXT_TOKEN = '<|endoftext|>'

def set_env(ENV):
    if ENV.lower() == 'yizhe':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif ENV.lower() == 'docker':
        pass
    elif ENV.lower() == 'scphilly' or ENV.lower() == 'rrphilly':
        pass
    elif ENV.lower() == 'yizhephilly':
        pass
    elif ENV.lower() == 'siqi':
        pass
    else:
        raise NotImplementedError('%s not implemented' % ENV)


def get_debug_argv(ENV):
    if ENV.lower() == 'yizhe':
        argv = [
                '--model_name_or_path', '/home/yizhe/ssd0/GPT-2/pretrained/117M',
                '--seed', '42',
                '--max_seq_length', '128',
                '--init_checkpoint', '/home/yizhe/ssd0/GPT-2/pretrained/117M/pytorch_model.bin',
                '--train_batch_size', '64',
                '--gradient_accumulation_steps', '4',
                '--eval_batch_size', '1',
                '--output_dir', '/home/yizhe/ssd0/GPT-2/outputs/GPT',
                '--normalize_data', 'True',
                '--fp16', 'False',
                '--lr_schedule', 'noam',
                '--tgt_token', 
                '--train_input_file', '/home/yizhe/ssd0/GPT-2/train.main.v1.batch1024.src_tgt.dev.5K.db',
                '--eval_input_file', '/home/yizhe/ssd0/GPT-2/train_data/dev.50.tsv',
        ]
    elif ENV.lower() == 'docker':
        argv = [
                '--model_name_or_path', '/go/home/yizhe/ssd0/GPT-2/pretrained/117M',
                '--seed', '42',
                '--max_seq_length', '256',
                '--init_checkpoint', '/go/home/yizhe/ssd0/GPT-2/pretrained/117M/pytorch_model.bin',
                '--train_batch_size', '32',
                '--gradient_accumulation_steps', '2',
                '--eval_batch_size', '32',
                '--output_dir', '/go/home/yizhe/ssd0/GPT-2/outputs/GPT',
                '--normalize_data', 'True',
                '--fp16', 'True',
                '--train_input_file', '/go/home/yizhe/ssd0/GPT-2/train_data/train.2k.db',
                '--eval_input_file', '/go/home/yizhe/ssd0/GPT-2/train_data/dev.2k.tsv',
        ]
    elif ENV.lower() == 'scphilly':
        argv = [
                '--model_name_or_path', '/philly/sc3/resrchvc/yizzhang/GPT/pretrained/117M',
                '--seed', '42',
                '--max_seq_length', '128',
                '--init_checkpoint', '/philly/sc3/resrchvc/yizzhang/GPT/pretrained/117M/pytorch_model.bin',
                '--train_batch_size', '1024',
                '--gradient_accumulation_steps', '2',
                '--eval_batch_size', '256',
                '--normalize_data', 'True',
                '--fp16', 'True',
                '--output_dir', os.environ['PHILLY_JOB_DIRECTORY'] + '/models',
                '--train_input_file', '/philly/sc3/resrchvc/yizzhang/GPT/train.main.v1.batch1024.src_tgt.4M.db',
                '--eval_input_file', '/philly/sc3/resrchvc/yizzhang/GPT/train.main.v1.batch1024.src_tgt.dev.5K.tsv',
        ]
    elif ENV.lower() == 'rrphilly':
        argv = [
                '--model_name_or_path', '/philly/rr2/msrlabspvc1/yizzhang/GPT/pretrained/117M',
                '--seed', '42',
                '--max_seq_length', '128',
                '--init_checkpoint', '/philly/rr2/msrlabspvc1/yizzhang/GPT/pretrained/117M/pytorch_model.bin',
                '--train_batch_size', '1024',
                '--gradient_accumulation_steps', '2',
                '--eval_batch_size', '256',
                '--normalize_data', 'True',
                '--fp16', 'True',
                '--output_dir', os.environ['PHILLY_JOB_DIRECTORY'] + '/models',
                '--train_input_file', '/philly/rr2/msrlabspvc1/yizzhang/GPT/train.main.v1.batch1024.src_tgt.4M.db',
                '--eval_input_file', '/philly/rr2/msrlabspvc1/yizzhang/GPT/train.main.v1.batch1024.src_tgt.dev.5K.tsv',
        ]
    elif ENV.lower() == 'yizhephilly':
        argv = [
                '--model_name_or_path', '/var/storage/shared/msrlabs/sys/jobs/application_1551393098021_7449/GPT-2/pretrained/117M',
                '--seed', '42',
                '--max_seq_length', '128',
                '--init_checkpoint', '/var/storage/shared/msrlabs/sys/jobs/application_1551393098021_7449/GPT-2/pretrained/117M/pytorch_model.bin',
                '--train_batch_size', '16',
                '--gradient_accumulation_steps', '2',
                '--eval_batch_size', '16',
                '--output_dir', '/var/storage/shared/msrlabs/sys/jobs/application_1551393098021_7449/GPT-2/outputs/GPT',
                '--normalize_data', 'True',

                '--train_input_file', '/var/storage/shared/msrlabs/sys/jobs/application_1551393098021_7449/GPT-2/train_data/train.2k.src,' \
                                      '/var/storage/shared/msrlabs/sys/jobs/application_1551393098021_7449/GPT-2/train_data/train.2k.tgt',

                '--eval_input_file', '/var/storage/shared/msrlabs/sys/jobs/application_1551393098021_7449/GPT-2/train_data/dev.2k.src,' \
                                     '/var/storage/shared/msrlabs/sys/jobs/application_1551393098021_7449/GPT-2/train_data/dev.2k.tgt',
        ]
    elif ENV.lower() == 'siqi':
        # GPT2_DATA_FOLDER = '/ssd/siqi/CSR/GPT'
        GPT2_DATA_FOLDER = '/convaistorage1/GPT'
        argv = [
                '--model_name_or_path', os.path.join(GPT2_DATA_FOLDER, 'pretrained/117M'),
                '--seed', '42',
                '--max_seq_length', '128',
                '--init_checkpoint', os.path.join(GPT2_DATA_FOLDER, 'pretrained/117M/pytorch_model.bin'),
                '--train_batch_size', '200',
                '--gradient_accumulation_steps', '1',
                '--eval_batch_size', '200',
                '--output_dir', os.path.join(GPT2_DATA_FOLDER, 'outputs'),
                '--normalize_data', 'True',
                '--fp16', 'True',
                '--loss_scale', '0',
                '--train_input_file', os.path.join(GPT2_DATA_FOLDER, 'train_data/train.main.v1.batch1024.src_tgt.200k.db'),
                '--eval_input_file', os.path.join(GPT2_DATA_FOLDER, 'train_data/dev.2k.tsv'),
        ]
    else:
        raise NotImplementedError('%s not implemented' % ENV)
    return argv

