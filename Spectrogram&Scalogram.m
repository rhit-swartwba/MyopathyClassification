%Time Frequency Representation
%Getdata using getdata function coded by emglab
%Example file: /Users/blaiseswartwood/Downloads/SF data/Myopathy/N2001M02BB/N2001M02BB01/N2001M02BB01.bin
%Example input: /Users/blaiseswartwood/Downloads/SF data/Myopathy/N2001M02BB/N2001M02BB01
userIn = input('File Path: ', 's')
for i2 = 51 : 75
   dir = '/N2001C01BB'
   number = string(i2)
   fType = '.bin'
   fileName = strcat(userIn,dir,number,dir,number,fType)
   [EMG, rate] = getdata (fileName);
   %Specify split
   numberSamples = 7500
   sampleOverlap =100
   %Create Matrix
   matrixFromVector = buffer(EMG,numberSamples,sampleOverlap)
   %Split Matrix
   for i = 1 : 35 
       Emg = matrixFromVector(:, i);


       %Spectrogram
	%Sampling Frequency
       fs = 24000
	%length of hamming window
       window = 100
	%number of overlapping samples in window
       noverlap = 80
	%number of sampling points to calculate discrete Fourier Transform
       nfft = 100
       spectrogram(Emg,hamming(window),noverlap,nfft, fs, 'yaxis')
       set(gcf, 'Position',  [224, 224, 224, 224])
%Save spectrogram
       fPath = '/Users/blaiseswartwood/Downloads/Created Images/Spectrogram'
       fName = 'M_P1_Sp_' + number + '_' +  string(i)  + '.jpg'
       saveas(gcf, fullfile(fPath, fName), 'jpeg');
       %CWT
       cwt(Emg, 24000)
       set(gcf, 'Position',  [224, 224, 224, 224])
%Save Scalogram
       fPath = '/Users/blaiseswartwood/Downloads/Created Images/Scalogram'
       fName = 'M_P1_Sc_' + number + '_' +  string(i)  + '.jpg'
       saveas(gcf, fullfile(fPath, fName), 'jpeg');
   end
end
