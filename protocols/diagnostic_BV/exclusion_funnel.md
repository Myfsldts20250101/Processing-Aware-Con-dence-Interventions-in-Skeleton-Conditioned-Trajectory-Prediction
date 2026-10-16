# S_BV diagnostic set (both-valid, pre-intervene)

- created: `2026-10-02T22:29:27+0800`
- T_B n=**15603** (content-diagnostic; NOT full moving)
- S_BV intersection n=**15160** (coverage=0.9716)
- sha=`6d25264ebab71b89c61755e8e970f54d3c0bb545d57722f7219c798309163f30`
- rule=`BV1` min_common_joints=3

## Per-seed ok_bv

- d42: ok_bv=15331 cov=0.9826 status={'ok_bv': 15331, 'no_candidate': 214, 'bv_fail_insufficient_common_joints': 58}
- d43: ok_bv=15332 cov=0.9826 status={'ok_bv': 15332, 'no_candidate': 214, 'bv_fail_insufficient_common_joints': 57}
- d44: ok_bv=15319 cov=0.9818 status={'ok_bv': 15319, 'no_candidate': 214, 'bv_fail_insufficient_common_joints': 70}
- d45: ok_bv=15322 cov=0.9820 status={'ok_bv': 15322, 'no_candidate': 214, 'bv_fail_insufficient_common_joints': 67}
- d46: ok_bv=15326 cov=0.9822 status={'ok_bv': 15326, 'no_candidate': 214, 'bv_fail_insufficient_common_joints': 63}

## Notes

- Protocol A (full moving) is not run by default; do not fill A cells with B/S_BV.
- No silent pool upgrade / self-match.
- E2 all_zero placeholder is reported, not dropped post-intervene.
