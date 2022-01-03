close all; clear; clc

N = 4*4096;

E0 = 1e-9; nu0 = 0;
E1 = 1; nu1 = 1/3;

K0 = E0 / (3 * (1 - 2 * nu0));
K1 = E1 / (3 * (1 - 2 * nu1));
G0 = E0 / (2 * (1 + nu0));
G1 = E1 / (2 * (1 + nu1));

C0 = E0 / (1 - nu0^2) * [1, nu0, 0; nu0, 1, 0; 0, 0, (1 - nu0) / 2];
C1 = E1 / (1 - nu1^2) * [1, nu1, 0; nu1, 1, 0; 0, 0, (1 - nu1) / 2];
C0_ = E0 / (1 - nu0^2) * [1, 1, (1 - nu0) / 2, nu0];
C1_ = E1 / (1 - nu1^2) * [1, 1, (1 - nu1) / 2, nu1];

S = rand(2 * N, 5);
Gxy = (G1 - G0) * S(:, 1) - G0;
Ex = (E1 - E0) * S(1:N, 2) - E0;
Ey = (E1 - E0) * S(N+1:end, 2) - E0;
vxy = 2 * S(:, 3) - 1;
vyx = 2 * S(:, 4) - 1;
q = pi / 2 * S(:, 5);
s = sin(q);
c = cos(q);
Ey_ = (vyx(1:N) ./ vxy(1:N)) .* Ex;
Ex_ = (vxy(N+1:end) ./ vyx(N+1:end)) .* Ey;
Ex = cat(1, Ex, Ex_);
Ey = cat(1, Ey_, Ey);
C11 = Ex ./ (1 - vxy .* vyx);
C22 = Ey ./ (1 - vxy .* vyx);
C33 = Gxy;
C12 = vyx .* Ex;
C = [C11.*c.^4 + C22.*s.^4 + 2*C12.*c.^2.*s.^2 + 4*C33.*c.^2.*s.^2, ...
     C22.*c.^4 + C11.*s.^4 + 2*C12.*c.^2.*s.^2 + 4*C33.*c.^2.*s.^2, ...
     C33.*c.^4 + C33.*s.^4 + C11.*c.^2.*s.^2 - 2*C12.*c.^2.*s.^2 ...
     + C22.*c.^2.*s.^2 - 2*C33.*c.^2.*s.^2, ...
     c.*s.*(C12.*c.^2 - C22.*c.^2 + 2*C33.*c.^2 + C11.*s.^2 - C12.*s.^2 - 2*C33.*s.^2), ...
     c.*s.*(C11.*c.^2 - C12.*c.^2 - 2*C33.*c.^2 + C12.*s.^2 - C22.*s.^2 + 2*C33.*s.^2), ...
     C12.*c.^4 + C12.*s.^4 + C11.*c.^2.*s.^2 + C22.*c.^2.*s.^2 - 4*C33.*c.^2.*s.^2] ./ ...
     (c.^2 + s.^2).^4;

flag1 = and(and(Ey > E0, Ey < E1), and(Ex > E0, Ex < E1));
flag2 = and(and(-sqrt(Ey ./ Ex) < vxy, vxy < sqrt(Ey ./ Ex)), ...
            and(-sqrt(Ex ./ Ey) < vyx, vyx < sqrt(Ex ./ Ey)));
flag3 = and(C(:, 3) < C1(3, 3), and(C(:, 1) < C1(1, 1), C(:, 2) < C1(2, 2)));
flag4 = - C(:,3).*C(:,6).^2 + 2*C(:,6).*C(:,5).*C(:,4) - C(:,2).*C(:,5).^2 - ...
    C(:,1).*C(:,4).^2 + C(:,1).*C(:,2).*C(:,3) > 0;
flag = and(and(flag3, flag4), and(flag1, flag2));

Ctars = C(flag, :);
% scatter3(C(flag, 4), C(flag, 5), C(flag, 6), '.'); axis image; grid on; hold on

% subplot(2, 2, 1);
% scatter(Ex(flag), Ey(flag), '.'); axis image; grid on
% xlabel('E_x'); ylabel('E_y');
% subplot(2, 2, 3);
% scatter(vxy(flag), vyx(flag), '.'); axis image; grid on
% xlabel('v_{xy}'); ylabel('v_{yx}');
% subplot(2, 2, [2, 4]);
% scatter3(Ex(flag), Ey(flag), vxy(flag), '.'); axis image; grid on
% xlabel('E_x'); ylabel('E_y'); zlabel('v_{xy}')

%%
% close all

% [~, Y] = pca(C(flag, :));

% Y = tsne(C(flag, :), ...
%     'verbos', 1, ...
%     'NumDimensions', 3, ...
%     'Exaggeration', 1, ...
%     'Perplexity', 2000, ...
%     'LearnRate', 1000, ...
%     'Standardize', true, ...
%     'Options', statset('MaxIter', 1000));
% scatter3(Y(:, 1), Y(:, 2), Y(:, 3), '.'); axis image; grid on
%%
% close all
% p = [5 25 50 100 200 500 1000 2000];
% kld = [1.5414 1.5531 1.3939 1.2516 0.9832 0.6498 0.4132 0.2064];
% plot(p, kld)