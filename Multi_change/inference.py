import cv2
import torch
from torch.utils import data
import argparse
import json
import os
import time
import numpy as np
from tqdm import tqdm

# Cümle bazlı metrik için NLTK
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from data.LEVIR_MCI import LEVIRCCDataset
from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import get_eval_score
from utils_tool.metrics import Evaluator

def main(args):
    """
    MCI-Net için Inference (Tahmin) Betiği
    """
    # 1. Kelime Haritasını (Vocab) Yükle
    vocab_path = os.path.join(args.list_path, args.vocab_file + '.json')
    with open(vocab_path, 'r') as f:
        word_vocab = json.load(f)
        
    # Index'ten kelimeye hızlı dönüşüm için sözlük oluştur
    idx_to_word = {v: k for k, v in word_vocab.items()}

    # 2. Modeli Yükle
    print("Model yükleniyor...")
    snapshot_full_path = args.checkpoint
    checkpoint = torch.load(snapshot_full_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(args.network)
    encoder_trans = AttentiveEncoder(train_stage=None, n_layers=args.n_layers,
                                          feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                          heads=args.n_heads, dropout=args.dropout)
    decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim,
                                      vocab_size=len(word_vocab), max_lengths=args.max_length,
                                      word_vocab=word_vocab, n_head=args.n_heads,
                                      n_layers=args.decoder_n_layers, dropout=args.dropout)

    encoder.load_state_dict(checkpoint['encoder_dict'])
    encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
    decoder.load_state_dict(checkpoint['decoder_dict'])

    # Modelleri GPU'ya al ve değerlendirme moduna geçir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()
    encoder_trans = encoder_trans.to(device).eval()
    decoder = decoder.to(device).eval()

    # 3. Dataloader'ı Ayarla
    test_loader = data.DataLoader(
        LEVIRCCDataset(args.data_folder, args.list_path, 'test', args.token_folder, args.vocab_file,
                       args.max_length, args.allow_unk),
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 4. Çıktı Klasörünü Hazırla
    os.makedirs(args.result_path, exist_ok=True)

    references = list()
    hypotheses = list()
    sentence_results = list()

    count = 0
    limit = args.limit  # İşlenecek örnek sayısı (varsayılan: 100)
    smooth_func = SmoothingFunction().method4

    with torch.no_grad():
        for ind, (imgA, imgB, seg_label, token_all, token_all_len, _, _, name) in enumerate(
                tqdm(test_loader, desc=f"TEST SPLIT - {limit} ÖRNEK DEĞERLENDİRİLİYOR")):
            
            if count >= limit:
                break

            # Cihaza gönder
            imgA = imgA.to(device)
            imgB = imgB.to(device)
            token_all = token_all.squeeze(0).to(device)

            # --- İnference (Forward) ---
            if encoder is not None:
                feat1, feat2 = encoder(imgA, imgB)
            feat1, feat2, seg_pre = encoder_trans(feat1, feat2)
            
            # Beam size = 1 için tahmin
            seq = decoder.sample(feat1, feat2, k=1)

            # --- Cümleleri Çözümleme (Decoding) ---
            img_token = token_all.tolist()
            
            # Referansları temizle (<START>, <END>, <NULL> atılır)
            img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}], img_token))
            references.append(img_tokens)

            # Tahmini temizle
            pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
            hypotheses.append(pred_seq)

            # --- Cümle Bazlı BLEU-4 Hesaplaması (NLTK) ---
            # Indexleri NLTK'nin anlayacağı metin formatına çeviriyoruz
            refs_words = [[idx_to_word.get(w, '<UNK>') for w in ref] for ref in img_tokens]
            hyp_words = [idx_to_word.get(w, '<UNK>') for w in pred_seq]
            
            ind_bleu4 = sentence_bleu(refs_words, hyp_words, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_func)
            
            sentence_results.append({
                "image_name": name[0],
                "sentence": " ".join(hyp_words),
                "bleu_4": round(ind_bleu4, 4)
            })

            count += 1

    # --- Genel Skoru Hesapla ---
    print(f"\nToplam işlenen benzersiz görüntü sayısı: {count}")
    
    try:
        # MCI modelinizin kendi utils kütüphanesi
        metrics = get_eval_score(references, hypotheses)
        overall_bleu_4 = metrics['Bleu_4']
    except Exception as e:
        print(f"Genel metrik hesaplanırken pycocoevalcap hatası: {e}. NLTK corpus_bleu kullanılıyor...")
        from nltk.translate.bleu_score import corpus_bleu
        corpus_refs = [[[idx_to_word.get(w, '<UNK>') for w in ref] for ref in refs] for refs in references]
        corpus_hyps = [[idx_to_word.get(w, '<UNK>') for w in hyp] for hyp in hypotheses]
        overall_bleu_4 = corpus_bleu(corpus_refs, corpus_hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_func)

    print(f"\n[GENEL METRİKLER]")
    print(f"Genel BLEU-4: {overall_bleu_4:.4f}")

    # --- JSON Çıktısını Hazırla ve Kaydet ---
    output_data = {
        "metadata": {
            "split": "TEST",
            "overall_bleu_4": round(overall_bleu_4, 4),
            "total_images_processed": count
        },
        "predictions": {}
    }

    # Her cümleyi ve skorunu JSON'a aktar
    for res in sentence_results:
        image_identifier = res["image_name"]
        output_data["predictions"][image_identifier] = {
            "sentence": res["sentence"],
            "bleu_4": res["bleu_4"]
        }

    save_path = os.path.join(args.result_path, f'test_{limit}_inference_results.json')
    
    with open(save_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nTahminler ve cümle bazlı BLEU-4 skorları kaydedildi: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCI-Net Inference Script')

    # Data parameters
    parser.add_argument('--sys', default='win', help='system win or linux')
    parser.add_argument('--data_folder', default='D:\Dataset\Caption\change_caption\Levir-MCI-dataset\images', help='folder with image files')
    parser.add_argument('--list_path', default='./data/LEVIR_MCI/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_MCI/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_MCI", help='base name shared by data files.')

    # Inference parametreleri
    parser.add_argument('--limit', type=int, default=100, help='Kaç adet görsel işleneceği (örn: 100)')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--checkpoint', default='./models_ckpt/MCI_model.pth', help='path to checkpoint')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--result_path', default="./predict_result/", help='path to save the result JSON')
    
    # Backbone & Model parameters
    parser.add_argument('--network', default='segformer-mit_b1', help='define the backbone encoder to extract features')
    parser.add_argument('--encoder_dim', type=int, default=512, help='the dimension of extracted features using backbone ')
    parser.add_argument('--feat_size', type=int, default=16, help='define the output size of encoder to extract features')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=512, help='embedding dimension')

    args = parser.parse_args()

    main(args)