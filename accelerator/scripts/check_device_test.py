import json
import sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1])
actual=np.fromfile(root/"state.out",dtype='<f4')
expected=np.fromfile(root/"state.expected",dtype='<f4')
np.testing.assert_allclose(actual,expected,rtol=5e-4,atol=5e-6)
c=np.fromfile(root/"control.out",dtype='<u4')
for lane,tokens in enumerate(json.loads((root/"tokens.expected.json").read_text())):
    r=c[lane*16:(lane+1)*16]
    assert r[0]==2, r
    np.testing.assert_array_equal(c[r[9]:r[9]+r[11]],tokens)
print('PASS: device C++ tokens and recurrent state match tiled Python emulator')
