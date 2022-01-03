function fem = defineParam(M, rmin, penal, E0, nu0, E1, nu1)

num_elems = M^2;
num_nodes = (M + 1)^2;
num_dofs = 2 * num_nodes;

% Node coordinates
[x, y] = meshgrid(linspace(-M/2, M/2, M + 1), ...
                  linspace(M/2, -M/2, M + 1));
nodes = [x(:), y(:)];

% DOFs
indices = reshape((1:num_nodes)', M + 1, M + 1);
elems = [reshape(indices(2:end, 1:end-1), num_elems, 1), ...
         reshape(indices(2:end, 2:end), num_elems, 1), ...
         reshape(indices(1:end-1, 2:end), num_elems, 1), ...
         reshape(indices(1:end-1, 1:end-1), num_elems, 1)];
edofs = 2 * kron(elems, ones(1, 2)) - repmat([1, 0], 1, 4);
iK = reshape(kron(edofs, ones(8, 1))', 64 * num_elems, 1);
jK = reshape(kron(edofs, ones(1, 8))', 64 * num_elems, 1);
iF = reshape(kron(edofs, ones(3, 1))', 3 * 8 * num_elems, 1);
jF = reshape(kron(repmat(1:3, num_elems, 1), ones(1, 8))',3 * 8 * num_elems,1);

% Base material
[Ke0, Fe0] = base_material(E0, nu0);
[Ke1, Fe1] = base_material(E1, nu1);

% Macro displacement
U0 = zeros(num_dofs, 3);
U0(1:2:end, 1) = nodes(:, 1);
U0(2:2:end, 2) = nodes(:, 2);
U0(1:2:end, 3) = nodes(:, 2)/2;
U0(2:2:end, 3) = nodes(:, 1)/2;

% Constraints
lsn = indices(2:end-1, 1)';
rsn = indices(2:end-1, end)';
tsn = indices(1, 2:end-1);
bsn = indices(end, 2:end-1);
bln = indices(end, 1);
brn = indices(end, end);
tln = indices(1, 1);
trn = indices(1, end);
s1n = cat(2, lsn, bsn);
s2n = cat(2, rsn, tsn);
vn = cat(2, bln, brn, tln, trn);
s1d = 2 * kron(s1n, ones(1, 2)) - repmat([1, 0], 1, length(s1n));
s2d = 2 * kron(s2n, ones(1, 2)) - repmat([1, 0], 1, length(s2n));
vd = 2 * kron(vn, ones(1, 2)) - repmat([1, 0], 1, length(vn));
iC = cat(2, 1:8, 9:(4 * (M - 1) + 8), 9:(4 * (M - 1) + 8));
jC = cat(2, vd, s1d, s2d);
sC = cat(2, ones(1, 8), ones(1, 4 * (M - 1)), -ones(1, 4 * (M - 1)));
C = sparse(iC, jC, sC, length(vd) + length(s1d), num_dofs);
num_const = size(C, 1);

% Filter
[x, y] = meshgrid(linspace((1 - M)/2, (M - 1)/2, M), ...
                  linspace((M - 1)/2, (1 - M)/2, M));
cc0 = [x(:), y(:)];
D = 2 * M * ones(num_elems);
for dx = [-M, 0, M]
    for dy = [-M, 0, M]
        cc = [x(:) + dx, y(:) + dy];
        D = min(D, pdist2(cc0, cc));
    end
end
H = sparse(max(0, rmin - D));
Hs = H * ones(num_elems, 1);
H = H/diag(Hs);

fem = struct('num_nodes', num_nodes, 'num_elems', num_elems, ...
             'num_dofs', num_dofs, 'num_const', num_const, ...
             'nodes', nodes, 'elems', elems, 'edofs', edofs, 'cc', cc0, ...
             'iK', iK, 'jK', jK, 'iF', iF, 'jF', jF, ...
             'Ke0', Ke0, 'Fe0', Fe0, 'Ke1', Ke1, 'Fe1', Fe1, ...
             'U0', U0, 'C', C, 'H', H, 'penal', penal);

function [Ke, Fe] = base_material(E, nu)
Ke = [  12 - 4*nu,   3*nu + 3, - 2*nu - 6,   9*nu - 3,   2*nu - 6, - 3*nu - 3,       4*nu,   3 - 9*nu
         3*nu + 3,  12 - 4*nu,   3 - 9*nu,       4*nu, - 3*nu - 3,   2*nu - 6,   9*nu - 3, - 2*nu - 6
       - 2*nu - 6,   3 - 9*nu,  12 - 4*nu, - 3*nu - 3,       4*nu,   9*nu - 3,   2*nu - 6,   3*nu + 3
         9*nu - 3,       4*nu, - 3*nu - 3,  12 - 4*nu,   3 - 9*nu, - 2*nu - 6,   3*nu + 3,   2*nu - 6
         2*nu - 6, - 3*nu - 3,       4*nu,   3 - 9*nu,  12 - 4*nu,   3*nu + 3, - 2*nu - 6,   9*nu - 3
       - 3*nu - 3,   2*nu - 6,   9*nu - 3, - 2*nu - 6,   3*nu + 3,  12 - 4*nu,   3 - 9*nu,       4*nu
             4*nu,   9*nu - 3,   2*nu - 6,   3*nu + 3, - 2*nu - 6,   3 - 9*nu,  12 - 4*nu, - 3*nu - 3
         3 - 9*nu, - 2*nu - 6,   3*nu + 3,   2*nu - 6,   9*nu - 3,       4*nu, - 3*nu - 3,  12 - 4*nu]...
         * E / (24 * (1 - nu^2));
Fe = [-2, -2*nu, nu - 1; -2*nu, -2, nu - 1
       2,  2*nu, nu - 1; -2*nu, -2, 1 - nu
       2,  2*nu, 1 - nu;  2*nu,  2, 1 - nu
      -2, -2*nu, 1 - nu;  2*nu,  2, nu - 1]...
       * E / (4 * (1 - nu^2));