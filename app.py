from flask import Flask, render_template, request, send_file
import tempfile
import librosa
import soundfile
import torch
from train import CycleGANTraining
import preprocess
import numpy as np

app = Flask(__name__)

# Define function to generate audio
def generate_audio(input_file):
    logf0s_normalization = './cache/logf0s_normalization.npz'
    mcep_normalization = './cache/mcep_normalization.npz'
    coded_sps_A_norm = './cache/coded_sps_A_norm.pickle'
    coded_sps_B_norm = './cache/coded_sps_B_norm.pickle'
    model_checkpoint = './model_checkpoint/'
    resume_training_at = './model_checkpoint/_CycleGAN_CheckPoint'
    #     resume_training_at = None

    validation_A_dir = './data/british/'
    output_A_dir = './converted_sound/british'

    validation_B_dir = './data/indian/'
    output_B_dir = './converted_sound/indian/'
    model = CycleGANTraining(logf0s_normalization=logf0s_normalization,
                                    mcep_normalization=mcep_normalization,
                                    coded_sps_A_norm=coded_sps_A_norm,
                                    coded_sps_B_norm=coded_sps_B_norm,
                                    model_checkpoint=model_checkpoint,
                                    validation_A_dir=validation_A_dir,
                                    output_A_dir=output_A_dir,
                                    validation_B_dir=validation_B_dir,
                                    output_B_dir=output_B_dir,
                                    restart_training_at=resume_training_at)
    x = model.loadModel('model_checkpoint/model3000.pth')[1]
    x.eval()
    mcep_normalization = np.load(mcep_normalization)
    coded_sps_A_mean = mcep_normalization['mean_A']
    coded_sps_A_std = mcep_normalization['std_A']
    coded_sps_B_mean = mcep_normalization['mean_B']
    coded_sps_B_std = mcep_normalization['std_B']
    logf0s_normalization = np.load(logf0s_normalization)
    log_f0s_mean_A = logf0s_normalization['mean_A']
    log_f0s_std_A = logf0s_normalization['std_A']
    log_f0s_mean_B = logf0s_normalization['mean_B']
    log_f0s_std_B = logf0s_normalization['std_B']
    num_mcep = 36
    sampling_rate = 16000
    frame_period = 5.0
    n_frames = 128
    wav, _ = librosa.load(input_file, sr=sampling_rate, mono=True)
    wav = preprocess.wav_padding(wav=wav,
                                    sr=sampling_rate,
                                    frame_period=frame_period,
                                    multiple=4)
    f0, timeaxis, sp, ap = preprocess.world_decompose(
        wav=wav, fs=sampling_rate, frame_period=frame_period)
    f0_converted = preprocess.pitch_conversion(f0=f0,
                                                mean_log_src=log_f0s_mean_A,
                                                std_log_src=log_f0s_std_A,
                                                mean_log_target=log_f0s_mean_B,
                                                std_log_target=log_f0s_std_B)
    coded_sp = preprocess.world_encode_spectral_envelop(
        sp=sp, fs=sampling_rate, dim=num_mcep)
    coded_sp_transposed = coded_sp.T
    coded_sp_norm = (coded_sp_transposed -
                        coded_sps_A_mean) / coded_sps_A_std
    coded_sp_norm = np.array([coded_sp_norm])

    if torch.cuda.is_available():
        coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()
    else:
        coded_sp_norm = torch.from_numpy(coded_sp_norm).float()

    coded_sp_converted_norm = x(coded_sp_norm)
    coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
    coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
    coded_sp_converted = coded_sp_converted_norm * \
                            coded_sps_B_std + coded_sps_B_mean
    coded_sp_converted = coded_sp_converted.T
    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
    decoded_sp_converted = preprocess.world_decode_spectral_envelop(
        coded_sp=coded_sp_converted, fs=sampling_rate)
    wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted,
                                                        decoded_sp=decoded_sp_converted,
                                                        ap=ap,
                                                        fs=sampling_rate,
                                                        frame_period=frame_period)
    # return wav_transformed
    soundfile.write('test1.wav',wav_transformed, 16000, subtype=None, endian=None, format=None, closefd=True)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    # Get uploaded file
    file = request.files['audio_file']

    # Save file to temporary directory
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        file.save(temp_file.name)
        temp_file.close()
        input_file = temp_file.name

    # Generate output audio file
    generate_audio(input_file)

    # Send output file back to user for playback
    return send_file('test1.wav', mimetype='audio/wav', as_attachment=True, download_name='generated.wav')

if __name__ == '__main__':
    app.run(debug=True)
