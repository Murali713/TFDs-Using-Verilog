

Select_Method={'Choi_Williams_Distribution','Choi_Williams_Distribution_Noise','Continues_Wavelet_Transform','Continues_Wavelet_Transform_Noise','Pseudo_Wigner_Ville_Distribution','Pseudo_Wigner_Ville_Distribution_Noise','Rihaczek_Distribution','Rihaczek_Distribution_Noise','Short_Time_Fourier_Transform','Short_Time_Fourier_Transform_Noise','Stockwell_Transform','Stockwell_Transform_Noise','Wigner_Ville_Distribution','Wigner_Ville_Distribution_Noise','EXIT'};
K=menu('WAVELET TYPE',Select_Method);

if(K==1)
cd Choi_Williams_Distribution_design
    final_run_without_noise

elseif(K==2)
cd Choi_Williams_Distribution_design
    final_run_with_noise
 
elseif(K==3)
cd Continues_Wavelet_Transform_design
    final_cwt_run_without_noise

elseif(K==4)
cd Continues_Wavelet_Transform_design
   final_cwt_run_with_noise
    
elseif(K==5)
cd Pseudo_Wigner_Ville_Distribution_design
final_run_without_noise

 elseif(K==6)
cd Pseudo_Wigner_Ville_Distribution_design
final_run_with_noise

elseif(K==7)
cd Rihaczek_Distribution_design
   final_run_without_noise  
  
elseif(K==8)
cd Rihaczek_Distribution_design
  final_run_with_noise   
  
  elseif(K==9)
cd Short_Time_Fourier_Transform_design
final_run_without_noise

 elseif(K==10)
cd Short_Time_Fourier_Transform_design
final_run_withnoise

elseif(K==11)
cd Stockwell_Transform_design
   final_stransform_run_without_noise  
  
elseif(K==12)
cd Stockwell_Transform_design
  final_stransform_run_with_noise  
  
  elseif(K==13)
cd Wigner_Ville_Distribution_design
   final_run_without_noise  
  
elseif(K==14)
cd Wigner_Ville_Distribution_design
  final_run_with_noise  
else
    return;
    
  
end