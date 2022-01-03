classdef InvHomoMMA < handle
    properties
        fem, Ctar, mma, plot_freq
        beta, eta
        xbar, xhat, Md, UA, CH
        xhat_best, c_best, best_iters
        flist, clist
    end
    methods
        function obj = InvHomoMMA(fem, Ctar, plot_freq)
            if nargin < 3
                obj.plot_freq = 0;
            else
                obj.plot_freq = plot_freq;
            end
            
            obj.fem = fem;
            obj.Ctar = Ctar;
            obj.beta = 1;
            obj.eta = 0.5;
            obj.best_iters = 0;
            obj.c_best = 1;
            
            m = 12;
            n = fem.num_elems;
            obj.mma = struct( ...
                'm', m, 'n', n, ...
                'xmin', zeros(n, 1), ...
                'xmax', ones(n, 1), ...
                'c', ones(m, 1), 'd', ones(m, 1), ...
                'a0', 1, 'a', zeros(m, 1), ...
                'raa0epx', 1e-6, 'raaeps', 1e-6*ones(m, 1), ...
                'epsimin', 1e-7);
        end
        
        function [f0val, fval] = evalObjConst(obj, x)
            % Filtering
            obj.xbar = obj.fem.H * x;
            obj.xhat = projectDensity(obj.xbar, obj.beta, obj.eta);
            obj.Md = 400 * mean(obj.xhat .* (1 - obj.xhat));

            % Evaluate objective
            f0val = mean(obj.xhat);
            
            % Homogenization
            [obj.CH, obj.UA] = computeHomo(obj.xhat, obj.fem);
            
            % Evaluate constraint
            fval = obj.CH([1, 5, 9, 8, 7, 4]) - obj.Ctar;
            fval = cat(2, fval, -fval)' / norm(obj.Ctar) - 1e-3;
        end
        
        function [df0dx, dfdx] = evalGradJac(obj, x)
            dxdx = deprojectDensity(obj.xbar, obj.beta, obj.eta);
            
            % Compute gradient
            df0dx = ones(obj.fem.num_elems, 1) / obj.fem.num_elems;
            df0dx = obj.fem.H * (df0dx .* dxdx);

            % Compute jacobian
            dCH = computeHomoGrad(obj.xhat, obj.UA, obj.fem);
            idx = [1, 5, 9, 8, 7, 4];
            dfdx = zeros(6, obj.fem.num_elems);
            for i = 1:6
                dfdx(i, :) = (obj.fem.H * (dCH{idx(i)} .* dxdx))';
            end
            dfdx = cat(1, dfdx, -dfdx) / norm(obj.Ctar);
        end
        
        function xval = optim(obj, x0, max_iter, algorithm)
            % Initialize
            xval = x0;
            [f0val, fval] = obj.evalObjConst(xval);
            [df0dx, dfdx] = obj.evalGradJac(xval);
            obj.flist = f0val;
            obj.clist = max(abs(fval));

            xold1 = xval;
            xold2 = xval;
            low = obj.mma.xmin;
            upp = obj.mma.xmax;

            outiter = 0;
            cond = true;
            while cond
                outiter = outiter + 1;
                
                if strcmp(algorithm, 'gcmma')
                    % Compute low, upp, raa0, raa
                    [low, upp, raa0, raa] = asymp( ...
                        outiter, obj.mma.n, xval, xold1, xold2, ...
                        obj.mma.xmin, obj.mma.xmax, low, upp, ...
                        obj.mma.raa0epx, obj.mma.raaeps, df0dx, dfdx);

                    % GCMMA subproblem
                    [xmma, ymma, zmma, lam, xsi, eta_, mu, zet, s, f0app, fapp] = ...
                        gcmmasub(obj.mma.m, obj.mma.n, obj.mma.epsimin, ...
                        xval, obj.mma.xmin, obj.mma.xmax, low, upp, ...
                        raa0, raa, f0val, df0dx, fval, dfdx, ...
                        obj.mma.a0, obj.mma.a, obj.mma.c, obj.mma.d);

                    % New evaluation
                    [f0valnew, fvalnew] = obj.evalObjConst(xmma);

                    % Check conservative
                    conserv = concheck(obj.mma.m, obj.mma.epsimin, ...
                        f0app, f0valnew, fapp, fvalnew);

                    % Inner loop
                    initer = 0;
                    if conserv == 0
                        while conserv == 0 && initer <= 15
                            initer = initer + 1;

                            [raa0, raa] = raaupdate( ...
                                xmma, xval, obj.mma.xmin, obj.mma.xmax, low, upp, ...
                                f0valnew, fvalnew, f0app, fapp, raa0, raa, ...
                                obj.mma.epsimin);

                            [xmma, ymma, zmma, lam, xsi, eta_, mu, zet, s, f0app, fapp] = ...
                                gcmmasub(obj.mma.m, obj.mma.n, obj.mma.epsimin, ...
                                xval, obj.mma.xmin, obj.mma.xmax, low, upp, ...
                                raa0, raa, f0val, df0dx, fval, dfdx, ...
                                obj.mma.a0, obj.mma.a, obj.mma.c, obj.mma.d);

                            [f0valnew, fvalnew] = obj.evalObjConst(xmma);

                            conserv = concheck(obj.mma.m, obj.mma.epsimin, ...
                                f0app, f0valnew, fapp, fvalnew);
                        end
                    end
                elseif strcmp(algorithm, 'mma')
                    % MMA subproblem
                    [xmma, ymma, zmma, lam, xsi, eta_, mu, zet, s, low, upp] = ...
                        mmasub(obj.mma.m, obj.mma.n, outiter, xval, ...
                        obj.mma.xmin, obj.mma.xmax, xold1, xold2, ...
                        df0dx, fval, dfdx, low, upp, ...
                        obj.mma.a0, obj.mma.a, obj.mma.c, obj.mma.d);
                end

                xold2 = xold1;
                xold1 = xval;
                xval = xmma;
                [f0val, fval] = obj.evalObjConst(xval);
                [df0dx, dfdx] = obj.evalGradJac(xval);

                % Check convergence
                dx = norm(xval - xold1) / norm(xold1);
                [~, kktnorm, ~] = ...
                    kktcheck(xmma, ymma, zmma, lam, xsi, eta_, mu, zet, s, ...
                    obj.mma.xmin, obj.mma.xmax, df0dx, fval, dfdx, ...
                    obj.mma.a0, obj.mma.a, obj.mma.c, obj.mma.d);
                b = obj.callback(outiter, f0val, fval, dx, kktnorm);
                cond = dx >= 1e-3 && kktnorm >= 5e-5 && outiter <= max_iter && b;
                if ~cond && max(abs(fval)) > 1e-3
                    cond = true;
                    obj.mma.c = 1.01 * obj.mma.c;
                    if outiter > max_iter
                        break
                    end
                end
            end
            fprintf('dx=%.3e\n', dx)
            fprintf('kkt=%.3e\n', kktnorm)
            fprintf('iter=%3d\n', outiter)
            fprintf('b=%d\n', b)
        end

        function b = callback(obj, iter, f, c, dx, kktnorm)
            b = true;
            
            % Record
            obj.flist = cat(1, obj.flist, f);
            obj.clist = cat(1, obj.clist, max(abs(c)));
            if obj.c_best < obj.clist(end)
                obj.best_iters = obj.best_iters + 1;
            else
                obj.c_best = min(obj.clist);
                obj.best_iters = 0;
                obj.xhat_best = obj.xhat;
            end
            
            if iter > 50
                df = abs(diff(obj.flist));
                if mean(df(end-9, end)) / obj.flist(end) < 1e-4 && max(abs(c)) < 1e-3
                    b = false;
                elseif obj.best_iters > 100
                    b = false;
                end
            end
            
            if mod(iter, 10) == 0 && iter > 50
                obj.beta = min(16, 1.1 * obj.beta);
            end
            
            % Monitor
            if mod(iter, obj.plot_freq) == 0 && obj.plot_freq > 0
                subplot(2, 2, 1); cla
                patch('faces', obj.fem.elems, 'vertices', obj.fem.nodes, ...
                    'facecolor', 'flat', 'edgecolor', 'none', ...
                    'facevertexcdata', obj.xhat);
                axis image;
                colormap(flipud(gray));
                caxis([0, 1])
                colorbar;
                title(sprintf('[%3d] f=%.3e, c=%.3e', iter, f, max(abs(c))))
                subplot(2, 2, 2);
                bar([obj.CH([1, 5, 9, 8, 7, 4]); obj.Ctar]')
                subplot(2, 2, 3); cla
                plot(obj.flist, 'r'); 
                title(sprintf('dx=%.3e, kkt=%.3e, Md=%.1f', dx, kktnorm, obj.Md))
                subplot(2, 2, 4); cla
                plot(obj.clist, 'b'); 
                title(sprintf('beta=%.1f, c_{best}=%.3e, it=%d', obj.beta, obj.c_best, obj.best_iters))
                drawnow;
            end
        end
    end
end