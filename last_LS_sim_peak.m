%{ 
    Filename : stepped_freq_sim2.m
    Date     : 2024.01.29
    Author   : Jong-Hee Kim
    Brief    : 코너리플렉터 배치 및 시뮬레이션
               시뮬레이션 코드 수정 사항 
                - range profile 각각 측정으로 변경
                - EP-based CLEAN 알고리즘 적용
                - rectangular window를 이용해 CNN에 입력
                - LS-estimation
                - 위치 측정에 더 좋은 효율을 가짐
%}

close all
clear 

%% 물리파라미터
c                = 3e8;                   % 진공에서의 빛의 속력 (m/s)
SNRdB            = 15;                    % 표적에 대한 신호대잡음비
global x_sky_net
[file,path] = uigetfile();
if file == 0
    return;
end
fullpath = sprintf('%s%s',path,file);
file_name = 'test';
mkdir(file_name);
x_sky_net = load(fullpath);

%% 레이더 파라미터(상수값)
start_freq       = 24e9;                  % 시작주파수 (Hz)
B                = 250e6;                 % 대역폭 (Hz)
Q                = 512;                    % 주파수샘플수 최대모호성거리 512 * c/2B

K               = 100;     % 시뮬레이션 최대 반복 횟수
Est_RadarPos    = zeros(3, K);
n_cluster_list  = zeros(K, 1);

    %% 레이더 파라미터(계산값)
    dst_freq    = start_freq + B;        % 종료주파수 (Hz)
    dr          = c / (2*B);             % 거리해상도 (m)
    axis_range  = 0 : dr : ((Q-1) * dr); % Range Profile의 거리축 생성
    
    %% 주파수 축 생성
    F               = linspace(start_freq, dst_freq, Q)';
    
    %% 레이더와 표적 위치 (3차원 공간에서 정의 시간에 따른 표적의 이동 속도 정의)
    RadarPos    = [25,110, 10]';
    TargetPos   = [20 20 1;
                    202 21 0;
                    19 202 0;
                    201 203 2]';% Unit(m)/ Target의 위치 행렬이 행 또는 열이 같은 숫자로 반복되면 안됨.ex) [x2 3 z; x2 3 z2; x3 3 z3; x4 3 z4];
    TargetAmp   = [4,  3,  5,  4];                % Unit(m)
    TargetAmp2  = [3,  4,  3,  2];
    TargetAmp3  = [2,  2,  3,  5];
    TargetAmp   = [TargetAmp,TargetAmp2,TargetAmp3];
    

    x1 = TargetPos(1,1); y1 = TargetPos(2,1); z1 = TargetPos(3,1);
    x2 = TargetPos(1,2); y2 = TargetPos(2,2); z2 = TargetPos(3,2);
    x3 = TargetPos(1,3); y3 = TargetPos(2,3); z3 = TargetPos(3,3);
    x4 = TargetPos(1,4); y4 = TargetPos(2,4); z4 = TargetPos(3,4);
    

    %% Target number 구분을 위한 추가 point target 설정
    xk1 = x1 - 12*cosd(45);    yk1 = y1 - 12*cosd(45);
    xk2 = x2 + 15*cosd(45);    yk2 = y2 - 15*cosd(45);
    xk3 = x3 - 21*cosd(45);    yk3 = y3 + 21*cosd(45);
    xk4 = x4 + 19*cosd(45);    yk4 = y4 + 19*cosd(45);
    xl1 = x1 - 25*cosd(45);    yl1 = y1 - 25*cosd(45);
    xl2 = x2 + 25*cosd(45);    yl2 = y2 - 25*cosd(45);
    xl3 = x3 - 25*cosd(45);    yl3 = y3 + 25*cosd(45);
    xl4 = x4 + 25*cosd(45);    yl4 = y4 + 25*cosd(45);

    TargetPos2 = [xk1, yk1, z1; xk2, yk2, z2; xk3, yk3, z3; xk4, yk4, z4]';
    TargetPos3 = [xl1, yl1, z1; xl2, yl2, z2; xl3, yl3, z3; xl4, yl4, z4]';
    TargetPos  = [TargetPos,TargetPos2,TargetPos3];

    R          = vecnorm(TargetPos - RadarPos, 2);%실제 레이더와 타겟의 거리 값
    e          = zeros(3,K);
   
%% 반복 진행
for r = 1:3

    for k = 1:K
        rn = [];
        plot_list_r = [];
        target = zeros(4,2);
        for tps = 1:4
            tg = 0;
            n_cluster = 0;
            Y = zeros(Q, 1);
           
            %% IQ 데이터 생성(주파수도메인) / EP_CLEAN 알고리즘 수행
            for rec = 0:2
                tp = tps+rec*4;
                Y = Y + TargetAmp(tp).*exp(-1i*4*pi.*F*R(tp)/c); %-j 2 pi f * (2r/c)
            end   
            Y = awgn(Y, SNRdB, 'measured');
            Yk = Y;
            
            [~,fd_idx] = maxk(abs(ifft(Y)),3);
               
            fdrp_idx = fd_idx*dr;
            % 비용 함수 정의
            for rec = 0:2
                tp = tps+rec*4;
                Jm = @(Am,Rl) sum(abs(Yk- Am.* exp(-1j*4*pi.*F.*Rl/c)).^2);
                
                % EP 알고리즘 매개변수 설정
                population_size    = 400;   % 개체 집단 크기
                max_generations    = 160;   % 최대 세대 수
                mutation_rate_Am   = 0.1;   % 돌연변이 비율
                mutation_rate_R    = 0.1;

                % 초기 개체 생성 (Am과 R을 랜덤하게 초기화)
                initial_population_Am = 0.4+((max(TargetAmp)*1.2)-0.4)*rand(1, population_size);
                initial_population_R  = (min(fdrp_idx)-7) + ((max(fdrp_idx)+7)-(min(fdrp_idx)-7)) * rand(1, population_size);
            
                % EP 알고리즘 실행
                current_population_Am = initial_population_Am;
                current_population_R  = initial_population_R;

                for generation = 1:max_generations
                    % 적합도 함수 계산
                    %fitness = Jm(current_population_Am, current_population_R);
                    
                    % 돌연변이 적용
                    mutation_Am = current_population_Am + mutation_rate_Am * randn(1, population_size);
                    mutation_R  = current_population_R  + mutation_rate_R  * randn(1, population_size);
                                                
                    % 돌연변이된 개체와 기존 개체를 모두 포함한 새로운 인구 생성
                    combined_Am = [current_population_Am mutation_Am];
                    combined_R  = [current_population_R mutation_R];

                    % 새로운 돌연변이 개체의 적합도 계산
                    combined_fitness = Jm(combined_Am, combined_R);
            
                    % 적합도에 따라 정렬하고 상위 population_size 개체 선택
                    [~, sorted_idx] = sort(combined_fitness);
                    selected_idx = sorted_idx(1:population_size);
                    selected_Am = combined_Am(selected_idx);
                    selected_R  = combined_R(selected_idx);
                    
                    % 다음 세대를 준비
                    current_population_Am = selected_Am;
                    current_population_R  = selected_R;
                end

                % 최종 결과
                [~, best_idx] = min(Jm(current_population_Am, current_population_R));
                best_Am(:,tp) = abs(current_population_Am(best_idx));
                best_R(:,tp) = abs(current_population_R(best_idx));
                
                Yk = Yk-best_Am(tp).*exp(-1j*4*pi.*F.*best_R(tp)/c);
                
            end
            
            B_A(:,tps) = best_Am([tps tps+4 tps+8]);
            B_R(:,tps) = best_R([tps tps+4 tps+8]);
            B_T        = [B_R(:,tps),B_A(:,tps)];
            B_T        = sortrows(B_T);
            B_A(:,tps) = B_T(:,2);
            B_R(:,tps) = B_T(:,1);
            newR(:,tps)= [R(tps);R(tps +4);R(tps+8)];
            newR(:,tps)= sortrows(newR(:,tps));
            
            
            %% Range Profile 생성
            
            x_win          = hamming(Q);
            rp             = ifft(Y.*x_win);          % 부엽 감소를 위한 윈도우 처리
            Qx10           = Q * 10;
            rpx10          = ifft(Y.*x_win./0.54, Qx10) .* 10;           % straddle loss 제거를 위한 Zero-Padding, Hamming윈도우의 평균값인 0.54로 인한 진폭 손실 보상
            axis_range_x10 = interp1(1:Q, axis_range, linspace(1, Q, Qx10));
            
            start_r       = floor(B_T(1,1)) - 5;
            end_r         = floor(B_T(3,1)) + 5;
            abs_sdiff     = abs(axis_range - start_r);
            abs_ediff     = abs(axis_range - end_r);
            [~,start_idx] = min(abs_sdiff);
            [~,end_idx]   = min(abs_ediff);
            
            r10_c     = rpx10(start_idx*10:end_idx*10);

            x_win     = rectwin(length(r10_c));
            rp_win    = r10_c .* x_win;
            axis_c    = axis_range_x10(start_idx*10:end_idx*10);

            
            mx_rp = max(abs(rpx10));
            %% Detection 처리
            rp_figure(tps) = figure(tps);
            plot(axis_c, abs(rp_win)/mx_rp,'k','LineWidth',2)
            ylim([0 1.5])
            xlim([axis_c(1) axis_c(end)])
            %xlabel("Range(m)",'FontSize',12);
            %ylabel("Amplitude",'FontSize',12);
            %set(gca,'FontSize',12);
            axis off;
            set(gca,'xtick',[],'ytick',[]);
            set(gca,'LooseInset',get(gca,'TightInset'));
            set(gca,'Box','off');
            cd(file_name)
            saveas(rp_figure(tps),['rp_figure',num2str(tps),'.jpg'])
            cd('..');
            resize_img  = imread(['C:\Users\User\Desktop\add제안서\stepped_freq_sim_file\final_sim_\test\rp_figure',num2str(tps),'.jpg']);
            resize_img2 = imresize(resize_img,[263,350]);
            cd(file_name)
            imwrite(resize_img2,['rp_figure',num2str(tps),'.jpg']);
            drawnow
        
            net = x_sky_net.net;
            new_img = imread(['C:\Users\User\Desktop\add제안서\stepped_freq_sim_file\final_sim_\test\rp_figure',num2str(tps),'.jpg']);
            label = classify(net,new_img); %표적인식 CNN이용해서 구분하기 (Target 1,2,3,4 중 하나로 구분)

            %% 가장 가까운 R을 기준으로 추출
            
            
            if label =='target1'
                tg = 1;
                target(1,1) = tg;
                target(1,2) = B_T(1,1);
            elseif label == 'target2'
                tg = 2;
                target(2,1) = tg;
                target(2,2) = B_T(1,1);
            elseif label == 'target3'
                tg = 3;
                target(3,1) = tg;
                target(3,2) = B_T(1,1);
            else
                tg = 4;
                target(4,1) = tg;
                target(4,2) = B_T(1,1);
            end
            
            
            cd('..');
            rp_figure(tps) = figure(tps);
            plot(axis_range_x10, abs(rpx10)/mx_rp,'k','LineWidth',2)
            hold on
            plot(B_R(:,tps), B_A(:,tps)/mx_rp,'.','MarkerSize', 10, 'MarkerEdgeColor', 'r')
            hold off
            legend('RangeProfile', 'CLEAN Result','FontSize',12)
            xlim([0 axis_range(end)])
            ylim([0 1.5])
            grid on
            xlabel('Range(m)','FontSize',12)
            ylabel('Amplitude & Detection','FontSize',12)
            set(gca,'FontSize',12)
            saveas(rp_figure(tps),['rp_figure_see',num2str(tps),'.jpg'])
            drawnow
            
            % drp_figure(tps) = figure(4+tps);
            % stem(newR(1,tps), ones(1, 1), 'ko')
            % grid on
            % xlabel('Range(m)')
            % ylabel('Detection(1/0)')
            % ylim([0 1.2])
            % xlim([0 axis_range(end)])
            % saveas(drp_figure(tps),['drp_figure',num2str(tps),'.jpg'])
            
        end

        %% MMSE 추정 시작
        x1 = TargetPos(1,1); y1 = TargetPos(2,1); z1 = TargetPos(3,1);
        x2 = TargetPos(1,2); y2 = TargetPos(2,2); z2 = TargetPos(3,2);
        x3 = TargetPos(1,3); y3 = TargetPos(2,3); z3 = TargetPos(3,3);
        x4 = TargetPos(1,4); y4 = TargetPos(2,4); z4 = TargetPos(3,4);
        xk1 = TargetPos(1,5);    yk1 = TargetPos(2,5); 
        xk2 = TargetPos(1,6);    yk2 = TargetPos(2,6); 
        xk3 = TargetPos(1,7);    yk3 = TargetPos(2,7);
        xk4 = TargetPos(1,8);    yk4 = TargetPos(2,8);
        xl1 = TargetPos(1,9);    yl1 = TargetPos(2,9);
        xl2 = TargetPos(1,10);   yl2 = TargetPos(2,10);
        xl3 = TargetPos(1,11);   yl3 = TargetPos(2,11);
        xl4 = TargetPos(1,12);   yl4 = TargetPos(2,12);

        d1 = R(:,1);
        d2 = R(:,2);
        d3 = R(:,3);
        d4 = R(:,4); %실제 거릿값
        
        dn1 = target(1,2);
        dn2 = target(2,2);
        dn3 = target(3,2);
        dn4 = target(4,2); %노이즈가 섞인 추정 거릿값
    
        X = [2*(x1-x2) 2*(y1-y2) 2*(z1-z2);
            2*(x1-x3) 2*(y1-y3) 2*(z1-z3);
            2*(x1-x4) 2*(y1-y4) 2*(z1-z4);
            2*(x2-x3) 2*(y2-y3) 2*(z2-z3);
            2*(x2-x4) 2*(y2-y4) 2*(z2-z4);
            2*(x3-x4) 2*(y3-y4) 2*(z3-z4)];
        rnk = rank(X);
        
        b = [(x1^2-x2^2)+(y1^2-y2^2)+(z1^2-z2^2)-(d1^2-d2^2);
            (x1^2-x3^2)+(y1^2-y3^2)+(z1^2-z3^2)-(d1^2-d3^2);
            (x1^2-x4^2)+(y1^2-y4^2)+(z1^2-z4^2)-(d1^2-d4^2);
            (x2^2-x3^2)+(y2^2-y3^2)+(z2^2-z3^2)-(d2^2-d3^2);
            (x2^2-x4^2)+(y2^2-y4^2)+(z2^2-z4^2)-(d2^2-d4^2);
            (x3^2-x4^2)+(y3^2-y4^2)+(z3^2-z4^2)-(d3^2-d4^2)];
        
        bn = [(x1^2-x2^2)+(y1^2-y2^2)+(z1^2-z2^2)-(dn1^2-dn2^2);
            (x1^2-x3^2)+(y1^2-y3^2)+(z1^2-z3^2)-(dn1^2-dn3^2);
            (x1^2-x4^2)+(y1^2-y4^2)+(z1^2-z4^2)-(dn1^2-dn4^2);
            (x2^2-x3^2)+(y2^2-y3^2)+(z2^2-z3^2)-(dn2^2-dn3^2);
            (x2^2-x4^2)+(y2^2-y4^2)+(z2^2-z4^2)-(dn2^2-dn4^2);
            (x3^2-x4^2)+(y3^2-y4^2)+(z3^2-z4^2)-(dn3^2-dn4^2)]; %노이즈 섞인 b
        
        t = (X'*X)\X'*b; %실제 좌표값
        tx = t(1,:);
        ty = t(2,:);
        tz = t(3,:);
        tn = (X'*X)\X'*bn; %노이즈가 섞인 예측 좌표값
        tnx = tn(1,:);
        tny = tn(2,:);
        tnz = tn(3,:);
    
        Est_RadarPos(1, k) = tnx;
        Est_RadarPos(2, k) = tny;
        Est_RadarPos(3, k) = tnz;
        
        
        e(r,k) = sqrt((tx- tnx)^2+(ty- tny)^2+(tz- tnz)^2); %오차값
        if k == K
            RMSE(r) = sqrt(mean(e(r,:).^2,2)) %RMSE 오차율
        end   



        %% 추정 위치 좌표 찍기
        
        es_figure(r) = figure(8+r);
        plot3(tx, ty, tz,'o','MarkerSize', 8, 'MarkerEdgeColor', 'r')
        hold on;
        plot3(Est_RadarPos(1, 1:k), Est_RadarPos(2, 1:k), Est_RadarPos(3, 1:k), '.','MarkerSize', 12, 'MarkerEdgeColor', 'b')
        plot3(x1,y1,z1,'^','MarkerSize', 8, 'MarkerEdgeColor', 'k')
        plot3(x2,y2,z2,'^','MarkerSize', 8, 'MarkerEdgeColor', 'k')
        plot3(x3,y3,z3,'^','MarkerSize', 8, 'MarkerEdgeColor', 'k')
        plot3(x4,y4,z4,'^','MarkerSize', 8, 'MarkerEdgeColor', 'k')
        plot3(xk1,yk1,z1,'^','MarkerSize', 8, 'MarkerEdgeColor', 'k')
        plot3(xk2,yk2,z2,'^','MarkerSize', 8, 'MarkerEdgeColor', 'k')
        plot3(xk3,yk3,z3,'^','MarkerSize', 8, 'MarkerEdgeColor', 'k')
        plot3(xk4,yk4,z4,'^','MarkerSize', 8, 'MarkerEdgeColor', 'k')
        plot3(xl1,yl1,z1,'^','MarkerSize', 8, 'MarkerEdgeColor', 'k')
        plot3(xl2,yl2,z2,'^','MarkerSize', 8, 'MarkerEdgeColor', 'k')
        plot3(xl3,yl3,z3,'^','MarkerSize', 8, 'MarkerEdgeColor', 'k')
        plot3(xl4,yl4,z4,'^','MarkerSize', 8, 'MarkerEdgeColor', 'k')
    
        grid on;
        axis([-50 250 -50 250 0 100])
        title(['SNR=',num2str(SNRdB),'dB, comparison of real position and estimated position']);
        xlabel("x(m)",'FontSize',12);
        ylabel("y(m)",'FontSize',12);
        zlabel("z(m)",'FontSize',12);
        legend('Real position','Estimated posiotion','Postion of target','Location','northeast','FontSize',12);
        view(-80, 20); %80,20
        hold off
        set(gca,'FontSize',12)
        saveas(es_figure(r),['es_figure',num2str(r),'.jpg'])
        drawnow
    end
    SNRdB = SNRdB - 5; %SNR 점점 감소
end
