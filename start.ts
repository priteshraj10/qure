import { spawn } from 'child_process';
import path from 'path';
import chalk from 'chalk';

class ProcessManager {
  private frontendProcess: any;
  private backendProcess: any;

  constructor() {
    this.setupProcessHandlers();
  }

  private setupProcessHandlers() {
    process.on('SIGINT', () => this.cleanup());
    process.on('SIGTERM', () => this.cleanup());
  }

  private cleanup() {
    console.log(chalk.yellow('\nShutting down processes...'));
    this.frontendProcess?.kill();
    this.backendProcess?.kill();
    process.exit(0);
  }

  private startProcess(command: string, args: string[], name: string, cwd: string) {
    const process = spawn(command, args, {
      stdio: 'pipe',
      shell: true,
      cwd: path.resolve(__dirname, cwd)
    });

    process.stdout.on('data', (data) => {
      console.log(chalk.cyan(`[${name}] `) + data.toString());
    });

    process.stderr.on('data', (data) => {
      console.error(chalk.red(`[${name} ERROR] `) + data.toString());
    });

    return process;
  }

  async start() {
    console.log(chalk.green('Starting Qure AI System...'));

    // Start backend
    this.backendProcess = this.startProcess(
      'python',
      ['-m', 'uvicorn', 'main:app', '--reload', '--port', '3000'],
      'Backend',
      './backend'
    );

    // Wait for backend to initialize
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Start frontend
    this.frontendProcess = this.startProcess(
      /^win/.test(process.platform) ? 'npm.cmd' : 'npm',
      ['run', 'dev'],
      'Frontend',
      './frontend'
    );

    console.log(chalk.green('\nQure AI System is running!'));
    console.log(chalk.blue('Frontend: http://localhost:3001'));
    console.log(chalk.blue('Backend: http://localhost:3000'));
    console.log(chalk.yellow('\nPress Ctrl+C to stop all processes\n'));
  }
}

new ProcessManager().start().catch(console.error); 