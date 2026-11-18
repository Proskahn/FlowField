"""Exercise the shell setup order without a server or an OpenFOAM installation."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


class NativeLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.script=Path(__file__).resolve().parents[1]/'scripts/check_native.sh'
        self.venv=self.root/'venv';self.foam=self.root/'foam';self.mpi=self.root/'mpi'
        for directory in (self.venv/'bin',self.foam/'bin',self.mpi):directory.mkdir(parents=True)
        # Model activate's PATH restoration, including an already active venv.
        (self.venv/'bin/activate').write_text(
            'PATH="$_OLD_VIRTUAL_PATH"\nexport PATH="$FLOWFIELD_VENV/bin:$PATH"\n')
        self.bashrc=self.foam/'bashrc'
        self.bashrc.write_text('''[[ $# == 0 ]] || return 21
[[ $PATH == "$FLOWFIELD_TEST_MPI:$FLOWFIELD_VENV/bin:"* ]] || return 22
export WM_PROJECT_VERSION="${FLOWFIELD_TEST_VERSION:-v2312}"
export PATH="$FLOWFIELD_TEST_FOAM/bin:$PATH"
''')
        record=self.root/'invocation.json'
        code='import json,os,sys; from pathlib import Path; Path(os.environ["FLOWFIELD_TEST_RECORD"]).write_text(json.dumps(sys.argv[1:])); sys.exit(int(os.environ.get("FLOWFIELD_TEST_EXIT",0)))'
        self.executable(self.venv/'bin/python',f'exec {shlex.quote(sys.executable)} -c {shlex.quote(code)} "$@"')
        self.executable(self.foam/'bin/wmake','exit 0')
        self.executable(self.foam/'bin/checkMesh','exit "${FLOWFIELD_TEST_CHECK_EXIT:-0}"')
        self.record=record
        self.env=dict(os.environ,FLOWFIELD_VENV=str(self.venv),FLOWFIELD_OPENFOAM_BASHRC=str(self.bashrc),
            FLOWFIELD_TEST_MPI=str(self.mpi),FLOWFIELD_TEST_FOAM=str(self.foam),
            FLOWFIELD_TEST_RECORD=str(record),FLOWFIELD_MPI_MODULE='test/mpi',_OLD_VIRTUAL_PATH='/usr/bin:/bin')

    def tearDown(self):self.tmp.cleanup()

    def executable(self,path,body):
        path.write_text('#!/bin/bash\n'+body+'\n');path.chmod(0o755)

    def launch(self,**env):
        # A module already marked loaded must be unloaded/reloaded after activate.
        setup='''module() {
    [[ "$2" == test/mpi ]] || return 23
    if [[ "$1" == load ]]; then export PATH="$FLOWFIELD_TEST_MPI:$PATH"; fi
}
export -f module
exec bash "$@"
'''
        return subprocess.run(['bash','-c',setup,'bash',str(self.script),'--problem','P008',
            '--timeout','14400','--output',str(self.root/'output with spaces')],
            env=dict(self.env,**env),capture_output=True,text=True,timeout=10)

    def test_setup_order_argument_forwarding_and_child_exit_code(self):
        result=self.launch(FLOWFIELD_TEST_EXIT='7')
        self.assertEqual(result.returncode,7,result.stdout+result.stderr)
        self.assertEqual(json.loads(self.record.read_text()),['-m','flowfield.stability','run',
            '--problem','P008','--timeout','14400','--output',str(self.root/'output with spaces'),
            '--backend','native'])
        self.assertIn(str(self.foam/'bin/checkMesh'),result.stdout)

    def test_missing_tool_stops_before_checker(self):
        (self.foam/'bin/checkMesh').unlink()
        result=self.launch()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Missing checkMesh',result.stderr)
        self.assertFalse(self.record.exists())
        self.assertFalse((self.root/'output with spaces').exists())

    def test_unusable_binary_or_wrong_version_stops_before_checker(self):
        for options in ({'FLOWFIELD_TEST_CHECK_EXIT':'127'},{'FLOWFIELD_TEST_VERSION':'v2406'}):
            with self.subTest(options=options):
                result=self.launch(**options)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('No cases started',result.stderr)
                self.assertFalse(self.record.exists())


if __name__=='__main__':unittest.main()
