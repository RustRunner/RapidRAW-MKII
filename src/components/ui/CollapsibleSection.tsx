import { useRef, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';
import clsx from 'clsx';
import Text from './Text';
import { TextVariants, TextWeights } from '../../types/typography';

interface CollapsibleSectionProps {
  children: any;
  isOpen: boolean;
  onContextMenu?: any;
  onToggle: any;
  title: string;
}

export default function CollapsibleSection({
  children,
  isOpen,
  onContextMenu,
  onToggle,
  title,
}: CollapsibleSectionProps) {
  const contentRef = useRef<HTMLDivElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    const content = contentRef.current;
    if (!wrapper || !content) {
      return;
    }

    const updateMaxHeight = () => {
      if (isOpen) {
        const contentHeight = content.scrollHeight;
        wrapper.style.maxHeight = `${contentHeight}px`;
      } else {
        wrapper.style.maxHeight = '0px';
      }
    };

    updateMaxHeight();

    const resizeObserver = new ResizeObserver(updateMaxHeight);
    resizeObserver.observe(content);

    return () => resizeObserver.disconnect();
  }, [isOpen]);

  return (
    <div className="bg-surface rounded-lg overflow-hidden shrink-0" onContextMenu={onContextMenu}>
      <div
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-card-active transition-colors duration-200"
        onClick={onToggle}
      >
        <Text variant={TextVariants.title} weight={TextWeights.normal}>
          {title}
        </Text>
        <ChevronDown
          className={clsx('text-accent transition-transform duration-300', { 'rotate-180': isOpen })}
          size={20}
        />
      </div>
      <div ref={wrapperRef} className="overflow-hidden transition-all duration-300 ease-in-out">
        <div className="px-4 pb-4" ref={contentRef}>
          {children}
        </div>
      </div>
    </div>
  );
}
