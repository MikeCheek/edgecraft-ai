import React, { useState, useEffect } from 'react';
import { SelectOption } from './constants';


// ---------------------------------------------------------------------------
// Reusable SelectOrCustom
// ---------------------------------------------------------------------------
// Renders a styled <select> from an options array. When the user picks the
// entry whose value === 'custom' an inline number <input> appears next to it.
//
// Props:
//   options   – array of { label: string; value: number | string }
//               (last entry should be { label: 'Custom...', value: 'custom' })
//   value     – current controlled value (number or string)
//   onChange  – (newValue: number | string) => void
//   disabled  – forwarded to both select and input
//   min       – optional forwarded to number input
//   max       – optional forwarded to number input
//   step      – optional forwarded to number input
//   className – optional extra className applied to the select
// ---------------------------------------------------------------------------

interface SelectOrCustomProps {
  options: SelectOption[];
  value: number | string;
  onChange: (val: number | string) => void;
  disabled?: boolean;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
}

function SelectOrCustom({
  options,
  value,
  onChange,
  disabled = false,
  min,
  max,
  step,
  className = '',
}: SelectOrCustomProps) {
  // Determine whether we are currently in custom mode
  const isPreset = options.some((o) => o.value !== 'custom' && o.value === value);
  const [isCustom, setIsCustom] = useState<boolean>(!isPreset);
  const [customValue, setCustomValue] = useState<string>(String(value));

  // Keep customValue in sync when value changes from outside
  useEffect(() => {
    const preset = options.some((o) => o.value !== 'custom' && o.value === value);
    if (preset) {
      setIsCustom(false);
    } else {
      setIsCustom(true);
      setCustomValue(String(value));
    }
  }, [value]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectCls =
    `w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm ` +
    `disabled:opacity-50 focus:border-purple-500 focus:outline-none ${className}`;

  const handleSelectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const raw = e.target.value;
    if (raw === 'custom') {
      setIsCustom(true);
      setCustomValue(String(value));
    } else {
      setIsCustom(false);
      const parsed = isNaN(Number(raw)) ? raw : Number(raw);
      onChange(parsed);
    }
  };

  const handleCustomChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setCustomValue(raw);
    const parsed = parseFloat(raw);
    if (!isNaN(parsed)) onChange(parsed);
  };

  const selectValue = isCustom
    ? 'custom'
    : options.some((o) => o.value === value)
      ? String(value)
      : 'custom';

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <select
        value={selectValue}
        onChange={handleSelectChange}
        disabled={disabled}
        className={selectCls}
        style={{ flex: '1 1 auto' }}
      >
        {options.map((opt) => (
          <option key={String(opt.value)} value={String(opt.value)}>
            {opt.label}
          </option>
        ))}
      </select>

      {isCustom && (
        <input
          type="number"
          value={customValue}
          onChange={handleCustomChange}
          disabled={disabled}
          min={min}
          max={max}
          step={step}
          className="px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm disabled:opacity-50 focus:border-purple-500 focus:outline-none"
          style={{ width: '100px', flex: '0 0 auto' }}
        />
      )}
    </div>
  );
}


export default SelectOrCustom;