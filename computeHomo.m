function [CH, UA] = computeHomo(x, fem)

xp = x.^fem.penal;

sK = fem.Ke1(:) * xp(:)' + fem.Ke0(:) * (1 - xp(:))';
K = sparse(fem.iK, fem.jK, sK(:), fem.num_dofs, fem.num_dofs);

sF = fem.Fe1(:) * xp(:)' + fem.Fe0(:) * (1 - xp(:))';
F = sparse(fem.iF, fem.jF, sF(:), fem.num_dofs, 3);

KC = [K, fem.C'; fem.C, zeros(fem.num_const)];
FB = [F; zeros(fem.num_const, 3)];
UL = KC \ FB;
U = UL(1:fem.num_dofs, :);

UA = fem.U0 - U;
CH = (UA' * K * UA) / fem.num_elems;

