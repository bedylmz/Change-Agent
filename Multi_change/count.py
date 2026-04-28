import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json
from tqdm import tqdm
from data.LEVIR_MCI import LEVIRCCDataset
from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *
from utils_tool.metrics import Evaluator

import torch

def count_parameters(model):
    """
    Modeldeki toplam ve eğitilebilir parametre sayısını hesaplar.
    """
    # Tüm parametrelerin toplamı
    total_params = sum(p.numel() for p in model.parameters())
    
    # Sadece 'requires_grad == True' olan (eğitilebilir) parametrelerin toplamı
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*40)
    print(f"Toplam Parametre Sayısı:      {total_params:,}")
    print(f"Eğitilebilir Parametre Sayısı: {trainable_params:,}")
    
    # Sıfıra bölme hatasını önlemek için kontrol
    if total_params > 0:
        print(f"Eğitilebilir Parametre Oranı:  %{(trainable_params / total_params) * 100:.4f}")
    print("="*40)
    
    return total_params, trainable_params

# Örnek Kullanım:
# from models.change_agent import ChangeAgentModel # (Reponuza göre import edin)
# model = ChangeAgentModel() 
# count_parameters(model)

class Trainer(object):
    def __init__(self, args):
        """
        Training and validation.
        """
        self.start_train_goal = args.train_goal
        self.args = args
        random_str = str(random.randint(10, 100))
        name = 'baseline_'+time_file_str() + f'_train_goal_{args.train_goal}_' + random_str
        self.args.savepath = os.path.join(args.savepath, name)
        self.args.savepath = os.path.join(args.savepath, name)
        if os.path.exists(self.args.savepath)==False:
            os.makedirs(self.args.savepath)
        self.log = open(os.path.join(self.args.savepath, '{}.log'.format(name)), 'w')
        print_log('=>datset: {}'.format(args.data_name), self.log)
        print_log('=>network: {}'.format(args.network), self.log)
        print_log('=>encoder_lr: {}'.format(args.encoder_lr), self.log)
        print_log('=>decoder_lr: {}'.format(args.decoder_lr), self.log)
        print_log('=>num_epochs: {}'.format(args.num_epochs), self.log)
        print_log('=>train_batchsize: {}'.format(args.train_batchsize), self.log)

        self.best_bleu4 = 0.4  # BLEU-4 score right now
        self.MIou = 0.4
        self.Sum_Metric = 0.4
        self.start_epoch = 0
        with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
            self.word_vocab = json.load(f)
        # Initialize / load checkpoint
        self.build_model()

        # Loss function
        self.criterion_cap = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_det = torch.nn.CrossEntropyLoss().cuda()

        # Custom dataloaders
        if args.data_name == 'LEVIR_MCI':
            self.train_loader = data.DataLoader(
                LEVIRCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
                batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
            self.val_loader = data.DataLoader(
                LEVIRCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

        self.index_i = 0
        self.hist = np.zeros((args.num_epochs*2 * len(self.train_loader), 5))
        # Epochs

        self.evaluator = Evaluator(num_class=3)

        self.best_model_path = None
        self.best_epoch = 0

    def build_model(self):
        args = self.args
        if args.train_stage == 's1':
            self.encoder = Encoder(args.network)
            self.encoder.fine_tune(args.fine_tune_encoder)
            self.encoder_trans = AttentiveEncoder(train_stage=args.train_stage, n_layers=args.n_layers,
                                                  feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                                  heads=args.n_heads, dropout=args.dropout)
            self.decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim,
                                              vocab_size=len(self.word_vocab), max_lengths=args.max_length,
                                              word_vocab=self.word_vocab, n_head=args.n_heads,
                                              n_layers=args.decoder_n_layers, dropout=args.dropout)

            fine_tune_capdecoder = True
        elif args.train_stage == 's2' and args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint)
            print('Load Model from {}'.format(args.checkpoint))
            # self.start_epoch = checkpoint['epoch'] + 1
            # self.best_bleu4 = checkpoint['bleu-4']
            self.decoder.load_state_dict(checkpoint['decoder_dict'])
            self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
            self.encoder.load_state_dict(checkpoint['encoder_dict'])
            # eval()
            self.encoder.eval()
            self.encoder_trans.eval()
            self.decoder.eval()
            # 各个modules 是否需要微调
            args.fine_tune_encoder = False
            self.encoder.fine_tune(args.fine_tune_encoder)
            self.encoder_trans.fine_tune(args.train_goal)
            fine_tune_capdecoder = False if args.train_goal == 0 else True
            self.decoder.fine_tune(fine_tune_capdecoder)
        else:
            # print('Error: checkpoint is None or stage=s1.')
            raise ValueError('Error: checkpoint is None.')

        # set optimizer
        self.encoder_optimizer = torch.optim.Adam(params=self.encoder.parameters(),
                                                  lr=args.encoder_lr) if args.fine_tune_encoder else None
        self.encoder_trans_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.encoder_trans.parameters()),
            lr=args.encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
            lr=args.decoder_lr) if fine_tune_capdecoder else None

        # Move to GPU, if available
        self.encoder = self.encoder.cuda()
        self.encoder_trans = self.encoder_trans.cuda()
        self.decoder = self.decoder.cuda()
        self.encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=5,
                                                                    gamma=1.0) if args.fine_tune_encoder else None
        self.encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_trans_optimizer, step_size=5,
                                                                          gamma=1.0)
        self.decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=5,
                                                                    gamma=1.0) if fine_tune_capdecoder else None

    def training(self, args, epoch):
        
        self.encoder.train()
        self.encoder_trans.train()
        self.decoder.train()  # train mode (dropout and batchnorm is used)
    
        if self.decoder_optimizer is not None:
            self.decoder_optimizer.zero_grad()
        self.encoder_trans_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()

        count_parameters(self.encoder)
        count_parameters(self.encoder_trans)
        count_parameters(self.decoder)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')

    # Data parameters
    parser.add_argument('--sys', default='win', help='system win or linux')
    parser.add_argument('--data_folder', default='D:\Dataset\Caption\change_caption\Levir-MCI-dataset\images',help='folder with data files')
    parser.add_argument('--list_path', default='./data/LEVIR_MCI/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_MCI/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_MCI",help='base name shared by data files.')

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint from stage s1, assert not None when train_stage=s2')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
    # Training parameters
    parser.add_argument('--train_goal', type=int, default=2, help='0:det; 1:cap; 2:two tasks')
    parser.add_argument('--train_stage', default='s1', help='s1: pretrain backbone under two loss;'
                                                                         ' s2: train two branch respectively')
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')
    parser.add_argument('--train_batchsize', type=int, default=64, help='batch_size for training')
    parser.add_argument('--num_epochs', type=int, default=250, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="./models_ckpt/")
    # backbone parameters
    parser.add_argument('--network', default='segformer-mit_b1', help='define the backbone encoder to extract features')
    parser.add_argument('--encoder_dim', type=int, default=512,
                        help='the dimension of extracted features using backbone ')
    parser.add_argument('--feat_size', type=int, default=16,
                        help='define the output size of encoder to extract features')
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=512, help='embedding dimension')
    args = parser.parse_args()

    trainer = Trainer(args)
    print('Starting Epoch:', trainer.start_epoch)
    print('Total Epoches:', trainer.args.num_epochs)

    if args.train_goal == 2:
        # First train both together, then train only change captioning, and finally train only change detection
        for goal in [2, 1, 0]:
            print_log(f'Current train_goal={goal}:\n', trainer.log)
            trainer.args.train_goal = goal
            if goal == 2:
                trainer.args.train_stage = 's1'
                trainer.args.checkpoint = None
                for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
                    trainer.training(trainer.args, epoch)
                    trainer.validation(epoch)
                    if epoch - trainer.best_epoch > 50:
                        trainer.start_epoch = trainer.best_epoch + 1
                        break
                    elif epoch == trainer.args.num_epochs - 1:
                        trainer.start_epoch = trainer.best_epoch + 1
                        trainer.args.num_epochs = trainer.start_epoch + args.num_epochs
            else:
                trainer.args.train_stage = 's2'
                trainer.args.checkpoint = trainer.best_model_path
                trainer.build_model()
                for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
                    trainer.training(trainer.args, epoch)
                    trainer.validation(epoch)
                    if trainer.args.train_goal == 1 and epoch - trainer.best_epoch > 50:
                        trainer.start_epoch = trainer.best_epoch + 1
                        trainer.args.num_epochs = trainer.start_epoch + trainer.args.num_epochs
                        break
                # trainer.args.num_epochs = trainer.start_epoch + trainer.args.num_epochs
    else:
        for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
            trainer.training(trainer.args, epoch)
            # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
