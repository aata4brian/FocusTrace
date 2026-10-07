import {describe,expect,it,vi} from 'vitest';
import {fireEvent,render,screen} from '@testing-library/react';
import {ConnectionStatus,formatTime,ProgressBar,SequenceBuilder} from './components';

describe('formatTime',()=>{
  it('formats and carries rounded seconds correctly',()=>{
    expect(formatTime(null)).toBe('—');
    expect(formatTime(-2)).toBe('00:00');
    expect(formatTime(0)).toBe('00:00');
    expect(formatTime(60.1)).toBe('01:01');
    expect(formatTime(119.1)).toBe('02:00');
  });
});

describe('ProgressBar',()=>{
  it('clamps semantic and visual progress',()=>{
    const {container}=render(<ProgressBar value={140}/>);
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow','100');
    expect(container.querySelector('.progress > div')).toHaveStyle({width:'100%'});
  });
});

describe('SequenceBuilder',()=>{
  it('moves blocks without changing membership',()=>{
    const onChange=vi.fn();
    render(<SequenceBuilder value={['A1','A2','B1','B2','C1','C2']} onChange={onChange}/>);
    fireEvent.click(screen.getByRole('button',{name:'Turunkan A1'}));
    expect(onChange).toHaveBeenCalledWith(['A2','A1','B1','B2','C1','C2']);
  });
  it('hides reordering controls when locked',()=>{
    render(<SequenceBuilder value={['A1','A2','B1','B2','C1','C2']} onChange={()=>{}} locked/>);
    expect(screen.queryByRole('button',{name:'Turunkan A1'})).toBeNull();
  });
});

describe('ConnectionStatus',()=>{
  it('renders synchronized and reconnecting states',()=>{
    const {rerender}=render(<ConnectionStatus online/>);
    expect(screen.getByText(/Tersinkron/)).toBeInTheDocument();
    rerender(<ConnectionStatus online={false}/>);
    expect(screen.getByText(/Menyambung ulang/)).toBeInTheDocument();
  });
});
