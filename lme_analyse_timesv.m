% Linear mixed-effects model class 
% https://au.mathworks.com/help/stats/linearmixedmodel-class.html



%% describtion 




%% pathways
loop=1;

%addpath('\\ad.monash.edu\home\User005\amah0011\Desktop\Insomnia_backup')
ins_loc=    '/home/ahmedm/ot95_scratch/Ahmed_Rest_insomniaproject/HCTSA_output_alloperations/ch6/Insomnia_subjects';
%'C:\Users\amah0011\Downloads\Honours_data\Insomnia'
ssm_loc= '/home/ahmedm/ot95_scratch/Ahmed_Rest_insomniaproject/HCTSA_output_alloperations/ch6/SSM_subjects';
control_loc= '/home/ahmedm/ot95_scratch/Ahmed_Rest_insomniaproject/HCTSA_output_alloperations/ch6/Control';




%% variables

group{1,1}='insomnia';
group{2,1}='ssm';
group{3,1}='GS';
n_conditions=3;
pathways= {ins_loc;ssm_loc;control_loc};
operations=7702;
sgs  =4;
stage=[0,1,2,3,5];

%% Organize data for analysis

 l1= 1:sgs
savename=sprintf('C3_M2_lme_HCTSA_stg%d_%d.mat',l1,loop);


sleep_stage= stage(1,l1+1); % exclude awake
 %
 full_mat=[];
 norm_mat=[];
  analyses=[];
   analyse=[];
   
opser=1; 
  
      
  for   con= 1:n_conditions 
  cd(pathways{con,1});
  names_sub=dir;
  n_subjects= size(names_sub,1)-2;
         for l2= 1: n_subjects
lx2= l2+2;
sub_name= names_sub(lx2,1).name;
load(sub_name)
info= regexp(sub_name,'\d*','Match');
ex=[2,2,1]';

sub_id= info{1,ex(con)};

                

[x,~]= find(sc==sleep_stage);
[subjects{1:size(x,1)}] = deal(sub_id); % clear variables at the end
Subjects_ID = subjects';  % clear variables at the end
%{'Sanchez';'Johnson';'Li';'Diaz';'Brown'};
%Age = [38;43;38;40;49];
%Smoker = logical([1;0;1;0;1]);

Feature_value= subject(x,opser);
Feature_values= nanmean(subject(x,opser));

full_mat= [full_mat;subject(x,:)];
          

norm_mat= [norm_mat;nanmean(subject(x,:))];

Subjects_IDs= Subjects_ID(1,1);

[Age{1:size(x,1),1}] = deal(age); 
%Age=Age';
Ages= Age(1,1);



[Gender{1:size(x,1),1}] = deal(gender); 
%Gender=Gender';
Genders= Gender(1,1);

condition_name= group{con,1};
[Condition{1:size(x,1)}] = deal(condition_name); 
Condition= Condition';
Conditions= Condition(1,1);

T= table(Subjects_ID,Feature_value,Condition,Age,Gender);
Ts= table(Subjects_IDs,Feature_values,Conditions,Ages,Genders);

analyses=[analyses;T];

analyse=[analyse;Ts];

clear T Subjects_ID Feature_values condition_name subjects Condition Feature_values Subjects_IDs Conditions Ts Age Gender

                    end

         
                end
                
      
         
  
         analyse.Ages= cell2mat(analyse.Ages);
analyse.Genders= cell2mat(analyse.Genders);

analyses.Age= cell2mat(analyses.Age);
analyses.Gender= cell2mat(analyses.Gender);
    
%end
%Height = [71;69;64;67;64];
%Weight = [176;163;131;133;119];
%BloodPressure = [124 93; 109 77; 125 83; 117 75; 122 80];

%% Linear mixed models Analysis
%Linear mixed-effects models are extensions of linear regression models for data that are collected and summarized in groups.
%These models describe the relationship between a response variable and independent variables, 
%with coefficients that can vary with respect to one or more grouping variables
%
for ops= 1: operations
    ops
try
    %iamhere
                  % analyses.Feautre_value=full_mat(:,ops);
                 analyse.Feature_values=norm_mat(:,ops);
                 analyses.Feature_value= full_mat(:,ops);
% adding first level Age and gender
%lme = fitlme(analyse,'Feature_values ~ 1 + Conditions');

lme = fitlme(analyse,'Feature_values ~ 1 + Ages + Genders');

 % using fixed effects variable only 
%lme3 = fitlme(analyse,'Feature_values ~ 1 + Conditions+ (1|Ages)+(1|Genders)');

lme2 = fitlme(analyse,'Feature_values ~ 1 + Conditions');

% % grouping epochs by client (using all segments of sleep stage)
%lme3 = fitlme(analyses,'Feature_value ~ 1 + Condition + (Condition|Subjects_ID)');
lme4 = fitlme(analyses,'Feature_value ~ 1 + Condition+ (Condition|Subjects_ID)');



% grouping epochs by client (using all segments of sleep stage) _+ adding
% random effects

%lme4 = fitlme(analyses,'Feature_value ~ 1 + Condition + (1|Age)+(1|Gender)+ (Condition|Subjects_ID)');
 lme3 = fitlme(analyses,'Feature_value ~ 1 + Age + Gender+(Age|Subjects_ID)+(Gender|Subjects_ID) ');


%% Comparing models and testing significance
% results = compare(lme,altlme) returns the results of a likelihood ratio test that compares the linear mixed-effects models lme and altlme. Both models must use the same response vector in the fit and lme must be nested in altlme for a valid theoretical likelihood ratio test. Always input the smaller model first, and the larger model second.
% compare tests the following null and alternate hypotheses:
% H0: Observed response vector is generated by lme.
% H1: Observed response vector is generated by model altlme.


whichmodel= compare(lme2,lme4);
whichmodel= single(whichmodel);
 pval = coefTest(lme2);
 
if pval  <  0.05



           if   whichmodel(2,8) < 0.05
    

                  if  lme3.LogLikelihood >  lme4.LogLikelihood
results= compare(lme3,lme4,'CheckNesting',true);   

results= single(results);

pval=results(2,8);
 Pvalues(1,ops)= pval;
 
 x= lme4.Coefficients;       %Intercept  %% ssm % gs
x(:,1)=[];
x= single(x);
info_operations{ops,1}= x;

                else


    x= lme4.Coefficients;       %Intercept  %% ssm % gs
x(:,1)=[];
x= single(x);

pval = coefTest(lme4);
Pvalues(1,ops)= pval;

 x= lme4.Coefficients;       %Intercept  %% ssm % gs
x(:,1)=[];
x= single(x);
info_operations{ops,1}= x;

                end
    

        else
    
    
   
results= compare(lme,lme2,'CheckNesting',true);

results= single(results);

pval=results(2,8);
 Pvalues(1,ops)= pval;
x= lme2.Coefficients;       %Intercept  %% ssm % gs
x(:,1)=[];
x= single(x);
info_operations{ops,1}= x;
    

        end

else

    x= lme2.Coefficients;       %Intercept  %% ssm % gs
x(:,1)=[];
x= single(x);
Pvalues(1,ops)= pval;

end
 %
%info_operations{ops,1}= x;

% results= compare(lme,lme2);
% r
 %   x= lme2.Coefficients;       %Intercept  %% ssm % gs
%x(:,1)=[];
%x= single(x);
%Pvalues(1,ops)= pval;esults1= single(results);
% 
% results= compare(lme2,lme4);
%  results2= single(results);
%   
%   
%  results= compare(lme,lme3);
%  results3= single(results);
  
%   
%   
% if results1(2,8) ==0
%     
%      [table,siminfo] = compare(lme,lme2,'nsim',1000,'CheckNesting',true);
%      table= single(table);
%      Pvalues(1,ops)= table(2,9);
%      
% elseif  results2(2,8) ==0
%     
%      [table,siminfo] = compare(lme,lme3,'nsim',1000,'CheckNesting',true);
% table= single(table);
%      Pvalues(1,ops)= table(2,9);
%   
% elseif results3(2,8) ==0
%         [table,siminfo] = compare(lme,lme4,'nsim',1000,'CheckNesting',true);
% table= single(table);
%      Pvalues(1,ops)= table(2,9);
%     
%     
% else
%    % Hypothesis test on fixed of linear mixed-effects model
%  
%   % p-value for the F-test on the fixed and/or random-effects coefficients of the linear mixed-effects model lme, returned as a scalar value.
% %pVal = coefTest(lme) returns the p-value for an F-test that all fixed-effects coefficients except for the intercept are 0. 
% 
% 
% pval = coefTest(lme);
% x= lme.Coefficients;       %Intercept  %% ssm % gs
% x(:,1)=[];
% x= single(x);
%  Pvalues(1,ops)= pval;
% info_operations{ops,1}= x;
% end
%     

   
%pval = coefTest(lme);

%x= lme.Coefficients;       %Intercept  %% ssm % gs
%x(:,1)=[];
%x= single(x);
 %Pvalues(1,ops)= pval;
%info_operations{ops,1}= x;

catch

fprintf('error')
end

 
end                 
cd('/home/ahmedm/ot95_scratch/Ahmed_Rest_insomniaproject/HCTSA_output_alloperations/ch6')
save(savename,'Pvalues','info_operations','-v7.3')




             
%          end 
% clear names
% 
%     end
%    
% end
