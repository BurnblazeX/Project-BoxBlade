import { file, section, ok, near, truthy } from './lib/harness.mjs';
import { cornerAO, faceAO, VOXEL_AO_MIN } from '../js/voxelao.js';

file('voxelao.test.mjs - static voxel AO for shadows off');

section('one corner');
ok('open corner is fully open', cornerAO(0, 0, 0), 1);
near('one side filled', cornerAO(1, 0, 0), 2 / 3);
near('only the diagonal', cornerAO(0, 0, 1), 2 / 3);
ok('both sides filled is enclosed, whatever the diagonal', cornerAO(1, 1, 0), 0);

section('across a face');
const open = () => 0;
ok('nothing around: no occlusion anywhere', faceAO(open, 0.5, 0.5), 1);
// A wall along one edge (i = -1 for every j): that edge darkens, the far
// edge does not.
const wall = (i, j) => (i === -1 ? 1 : 0);
truthy('darker at the wall', faceAO(wall, 0.05, 0.5) < faceAO(wall, 0.95, 0.5));
near('fully open on the far side', faceAO(wall, 1, 0.5), 1);
// An inner corner: walls along two edges meet.
const corner = (i, j) => (i === -1 || j === -1 ? 1 : 0);
near('the inner corner hits the floor value', faceAO(corner, 0, 0), VOXEL_AO_MIN);
truthy('and it is the darkest point', faceAO(corner, 0, 0) < faceAO(corner, 0.5, 0) &&
       faceAO(corner, 0.5, 0) < faceAO(corner, 0.5, 0.5));
