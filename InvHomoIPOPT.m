classdef InvHomoIPOPT < handle
    properties
        fem, Ctar, plot_freq
        beta, eta
        xbar, xhat, Md, UA, CH
        vf_scale, delC_scale
        flist
    end
    methods
        function obj = InvHomoIPOPT(fem, Ctar, plot_freq)
            if nargin < 3
                obj.plot_freq = 0;
            else
                obj.plot_freq = plot_freq;
            end
            
            obj.fem = fem;
            obj.Ctar = Ctar;
            obj.beta = 1;
            obj.eta = 0.5;
        end
        
        function f = objective(obj, x)
            % Filtering
            obj.xbar = obj.fem.H * x;
            
            % Projection
            obj.xhat = projectDensity(obj.xbar, obj.beta, obj.eta);
            obj.Md = 400 * mean(obj.xhat .* (1 - obj.xhat));

            % Evaluate volume fraction
            vf = mean(obj.xhat);
            if isempty(obj.vf_scale)
                obj.vf_scale = 1 / vf;
            end
            
            % Homogenization
            [obj.CH, obj.UA] = computeHomo(obj.xhat, obj.fem);
            
            % Evaluate constraint
            delC = obj.CH([1, 5, 9, 8, 7, 4]) - obj.Ctar;
            delC = sum(delC.^2);
            if isempty(obj.delC_scale)
                obj.delC_scale = 100 / delC;
            end
            
            f = obj.delC_scale * delC + obj.vf_scale * vf;
        end
        
        function g = gradient(obj, x)
            % Compute gradient
            dxdx = deprojectDensity(obj.xbar, obj.beta, obj.eta);
            dvf = ones(obj.fem.num_elems, 1) / obj.fem.num_elems;
            dvf = obj.fem.H * (dvf .* dxdx);
            
            % Compute jacobian
            dxdx = deprojectDensity(obj.xbar, obj.beta, obj.eta);
            dCH = computeHomoGrad(obj.xhat, obj.UA, obj.fem);
            delC = obj.CH([1, 5, 9, 8, 7, 4]) - obj.Ctar;
            idx = [1, 5, 9, 8, 7, 4];
            ddelC = zeros(6, obj.fem.num_elems);
            for i = 1:6
                ddelC(i, :) = (obj.fem.H * (dCH{idx(i)} .* dxdx))';
            end
            ddelC = 2 * delC * ddelC;
            g = obj.delC_scale * ddelC' + obj.vf_scale * dvf;
        end
        
        function x = optim(obj, x0, max_iter)
            % Initialize
            f0 = obj.objective(x0);
            obj.flist = f0;
            
            % Options
            options.lb = zeros(obj.fem.num_elems, 1);
            options.ub = ones(obj.fem.num_elems, 1);
            options.ipopt.hessian_approximation = 'limited-memory';
            options.ipopt.max_iter = max_iter;
            options.ipopt.acceptable_tol = 1e-2;
            options.ipopt.print_level = 0;
            
            % Functions
            funcs.objective         = @obj.objective;
            funcs.gradient          = @obj.gradient;
            funcs.iterfunc          = @obj.callback;
            
            % Strat optimization
            x = x0;
            while true
                x = ipopt(x, funcs, options);
                if obj.Md > 15
                    obj.beta = min(16, 2 * obj.beta);
                else
                    break
                end
            end
        end

        function b = callback(obj, iter, f, x)
            b = true;
            
            % Record
            obj.flist = cat(1, obj.flist, f);
            
%             if mod(iter, 10) == 0 && iter > 50 && obj.Md > 15
%                 obj.beta = min(8, 1.1 * obj.beta);
%             end
            
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
                title(sprintf('[%3d] f=%.3e', iter, f))
                subplot(2, 2, 2);
                bar([obj.CH([1, 5, 9, 8, 7, 4]); obj.Ctar]')
                subplot(2, 2, [3, 4]); cla
                semilogy(obj.flist, 'r');
                title(sprintf('Md=%.2f, beta=%.1f', obj.Md, obj.beta))
                drawnow;
            end
        end
    end
end