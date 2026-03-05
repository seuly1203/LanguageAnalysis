
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq,\
    AutoProcessor, AutoTokenizer, AutoModel, pipeline, \
    WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor,\
    AutomaticSpeechRecognitionPipeline
from peft import PeftModel, PeftConfig

import joblib
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from spacy.lang.sv import Swedish
import stanza
from jiwer import wer
from datetime import datetime
import jiwer

# import argparse
import warnings
warnings.filterwarnings('ignore')

from data_loader import AudioDataset
from functions import transform, get_sentence_embedding, spacy_segmentation, analyze_speech
from lr_train import gen_train_data


class Pipeline():
    def __init__(self, load_dataset=False, lr_train=False):
        self.lr_train = lr_train  
        
        if load_dataset: # LR train (w/ref) / WER calculation (train, test folder--> exclude) (w/ref) / only inference (test folder) (w/o ref)
            self.dataset = AudioDataset()

        self.embed_model, self.embed_tokenizer = self.load_embed_model()
        self.w_model, self.processor = self.load_whisper()
        self.lora_model, self.lora_tokenizer, self.lora_feature_extractor = self.load_whisper_lora()

        if self.lr_train:
            self.train_dataset = AudioDataset(ref=True, lr_train=self.lr_train)
            lr_model_id = self.train_lr_model()
            self.lr_model = self.load_lr_model(lr_model_id)
        else:
            self.lr_model = self.load_lr_model()

        self.spc_nlp = self.get_setencizer()
        self.stz_nlp = stanza.Pipeline(lang='sv', processors='tokenize, pos, lemma')

    def load_whisper(self):
        # transcription
        whisper_id = "KBLab/kb-whisper-large"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper_id, torch_dtype=torch_dtype, use_safetensors=True, cache_dir="cache"
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(whisper_id)

        return model, processor
    
    def load_whisper_lora(self):
        peft_model_id = "syi2m/whisper-lora"
        language = "sv"
        task = "transcribe"
        peft_config = PeftConfig.from_pretrained(peft_model_id)
        model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype="auto",
            device_map={"":0}
        )

        model = PeftModel.from_pretrained(model, peft_model_id)

        #
        model = model.merge_and_unload()
        #
        tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
        processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
        feature_extractor = processor.feature_extractor
        # forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)

        return model, tokenizer, feature_extractor
    
    def load_embed_model(self):
        # text embedding
        embedding_id = "KB/bert-base-swedish-cased"
        embed_tokenizer = AutoTokenizer.from_pretrained(embedding_id)
        embed_model = AutoModel.from_pretrained(embedding_id)

        return embed_model, embed_tokenizer

    def load_lr_model(self, model_name='logistic_model.pkl'):
        lr_model = joblib.load(f"./models/{model_name}") 
        return lr_model
    
    def get_setencizer(self,):
        # segmentation 
        spc_nlp = Swedish()
        spc_nlp.add_pipe("sentencizer")
        
        return spc_nlp

    def get_dataset(self):
        if not self.dataset:
            raise Exception()
        return self.dataset
    

    def train_lr_model(self):
        if not self.train_dataset:
            raise Exception()
        
        ref_data_path = self.train_dataset.get_ref_path()
        X_train, y_train, X_test, y_test = gen_train_data(ref_data_path, self.embed_tokenizer, self.embed_model, test_size=0.2)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        print("Logistic Regression Accuracy:", clf.score(X_test, y_test))

        x = datetime.now()
        date_str = x.strftime("%d_%m_%y_%X")
        model_name = f'lr_{date_str}.pkl'

        # save model
        joblib.dump(clf, f'./models/{model_name}')

        return model_name


    def speaker_assignment(self, transcript):
        sentences = spacy_segmentation(transcript, self.spc_nlp)

        kids_speech = []
        for _, s in enumerate(sentences): 
            embed = get_sentence_embedding(s, self.embed_tokenizer,self.embed_model)
            lr_pred = self.lr_model.predict([embed])
            lr_prob = self.lr_model.predict_proba([embed])
            prob = lr_prob[0][lr_pred][0]

            if lr_pred == 1:
                pid = "Kid"
                kids_speech.append(s)

            else:
                pid = "Teacher"
            print(f"[{pid} ({prob:.2f})]: {s}")
        print()

        return kids_speech

    def wer_calculation(self, ref, hyp, print_result=False):
        ref_transformed = transform(ref)
        text_transformed = transform(hyp)

        wer_score = wer(ref_transformed, text_transformed)

        if print_result:
            print(ref_transformed)
            print(text_transformed)
            print(wer_score)
            print()

        return wer_score

    def transcribe_base(self, aud_path):
        # ##### transcription v1
        # audio, sample_rate = sf.read(aud_path)
        # inputs = self.processor(audio, sampling_rate=sample_rate,
        #                         return_tensors="pt",
        #                         chunk_lenghs=80,
        #                         )
        # forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="sv", task="transcribe")
        # inputs["input_features"] = inputs["input_features"].half().to(device)
        # with torch.no_grad():
        #     outputs = self.w_model.generate(inputs.input_features,
        #                                     return_timestamps=True,
        #                                     forced_decoder_ids=forced_decoder_ids
        #                                     )

        # time_stamps = self.processor.tokenizer.decode(outputs[0], output_offsets=True)
        # word_timestamps = time_stamps["offsets"]

        # transcript = ''
        # for wt in word_timestamps:
        #     transcript += ' ' + wt['text']
        
        ##### transcription v2
        generate_kwargs = {
            "task": "transcribe",
            "language": "sv",
            # "compression_ratio_threshold":2, # 1.35
            # "temperature":0.2, # 0.0
            # "logprob_threshold": -1.0,
            # "no_speech_threshold": 0.4, #  0.6
            }
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.w_model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps=True
        )

        res = pipe(aud_path,
                   chunk_length_s=80,
                   generate_kwargs=generate_kwargs)
        
        transcript = res['text']


        return transcript

    def transcribe_lora(self, aud_path):
        pipe = AutomaticSpeechRecognitionPipeline(model=self.lora_model,
                                          tokenizer=self.lora_tokenizer, feature_extractor=self.lora_feature_extractor,
                                          # chunk_length_s=80,
                                          # compression_ratio_threshold=1.35,
                                          # log_prob__threshold=0.0
                                          )
        
        with torch.amp.autocast('cuda'):
            transcript = pipe(aud_path,
                              return_timestamps=True)['text']
        remove_white_space = jiwer.Compose([
                                            jiwer.RemoveMultipleSpaces(),
                                            jiwer.Strip(),
                                            ])
                                            
        transcript = remove_white_space(transcript)
        
        return transcript

    #
    def generate_with_confidence(self, model, inputs):
      inputs["input_features"] = inputs["input_features"].half().to("cuda")
      with torch.no_grad():
        output = model.generate(
          **inputs,
          return_dict_in_generate=True,
          output_scores=True,
          max_new_tokens=100,
        )

      tokens = output.sequences[0]
      text = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
      scores = output.scores  # list of tensors
      confidences = [torch.softmax(score, dim=-1).max().item() for score in scores]
      avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

      return text, avg_conf

    def inference_with_confidence(self, aud_path):
      waveform, sr = torchaudio.load(aud_path)
      inputs =self.processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")

      text_base, conf_base = self.generate_with_confidence(self.w_model, inputs)
      text_lora, conf_lora = self.generate_with_confidence(self.lora_model, inputs)

      # final_text = text_base if conf_base > conf_lora else text_lora
      final_text = self.transcribe_base(aud_path) if conf_base > conf_lora else self.transcribe_lora(aud_path)
      print(final_text)

      return final_text

    #

    def run_inference(self, aud_path):
        transcript = self.inference_with_confidence(aud_path)
        kids_speech = self.speaker_assignment(transcript) # print
        analyze_speech(kids_speech, self.stz_nlp)


    def run_inference_folder(self, data_dir):
        ds = AudioDataset(data_dir=data_dir)
        for _, aud_path in enumerate(ds):
          self.run_inference(aud_path)


    def run_inference_wer(self, ref='convo'):
        ds = AudioDataset(ref=ref)        
        wer_scores = []
        
        for i, (ds_aud_path, aud_label) in enumerate(ds):
                                  
            base_transcript = self.transcribe_base(ds_aud_path)
            lora_transcript = self.transcribe_lora(ds_aud_path)
            base_wer = self.wer_calculation(ref=aud_label, hyp=base_transcript)
            lora_wer = self.wer_calculation(ref=aud_label, hyp=lora_transcript)
            if lora_wer <= base_wer:
                wer_scores.append(lora_wer)
            else:
                wer_scores.append(base_wer)
        avg_wer = sum(wer_scores)/len(ds)
        print(f'Average WER: {avg_wer}')

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    pipe = Pipeline(load_dataset=True, lr_train=False)
    # pipe = Pipeline(load_dataset=True, lr_train=True)
    # pipe.run_inference("./data/test/0a665da1-0a23-45e6-b585-85d4c33ae182.wav")
    pipe.run_inference_folder("test") 
    # pipe.run_inference_wer(ref='blob') # calculate WER with reference data




